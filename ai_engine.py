"""
ai_engine.py - Production AI request engine for Bookiee AI.

Architecture decisions:
  1. CONSOLIDATION: analyze = 1 unified JSON call (was 3 separate calls).
     Reduces free-tier burn from 3 RPM to 1 RPM for the most common action.
  2. SINGLETON CLIENT: @st.cache_resource creates the Gemini client once
     globally, never recreated on reruns.
  3. RESPONSE CACHING: @st.cache_data with doc_hash key. Same document =
     zero redundant API calls across all reruns, tabs, and interactions.
  4. SMART RETRY:
       429 RESOURCE_EXHAUSTED -> NO retry. Show countdown, respect quota.
       503 UNAVAILABLE        -> Retry 2x with 15s / 30s backoff.
       Other transient        -> Retry once after 3s.
  5. RATE LIMITER: enforces 13s minimum gap between calls (free tier: ~4.5 RPM).
     Displays a short countdown rather than silently blocking.
  6. DEEP DIVES: lazy, button-triggered. Never execute automatically on expander
     open. Results stored in session state and cached by @st.cache_data.
  7. OBSERVABILITY: every call is logged with timing and outcome.
  8. USER ERRORS: raw API exceptions are never shown. All errors are translated
     to friendly messages.
"""

import streamlit as st
import time
import json
from typing import Any, Generator

from google import genai
from google.genai import types

import state
import prompts
from chunker_utils import chunk_doc, MAX_CHARS

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL         = "gemini-2.5-flash"
FREE_TIER_RPM = 4.5          # conservative; actual limit is 5 RPM
CALL_GAP      = 60 / FREE_TIER_RPM   # ~13 s minimum between calls
QUOTA_WAIT    = 62           # seconds to display after a real 429
BACKOFF_503   = [15, 30]     # retry wait times for 503 errors


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _get_client():
    """
    Singleton Gemini client.
    @st.cache_resource caches the return value globally across ALL reruns and
    sessions — the client is created exactly once per server process.
    Returns None if no API key is configured (handled gracefully in callers).
    """
    key = st.secrets.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=key) if key else None


def _require_client():
    c = _get_client()
    if c is None:
        st.error("Add GEMINI_API_KEY in Settings -> Secrets.")
        st.stop()
    return c


# ══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════
def _enforce_rate_limit() -> None:
    """
    Session-level rate limiter. Each user session tracks its own last-call
    timestamp — no shared state between users.

    Shows a short countdown (never silent blocking) so users understand
    why there is a brief pause.
    """
    now  = time.time()
    last = st.session_state.get(state.LAST_AI_TS, 0.0)
    wait = CALL_GAP - (now - last)
    if wait > 0.8:
        ph = st.empty()
        for t in range(int(wait), 0, -1):
            ph.caption(f"Pacing requests... {t}s")
            time.sleep(1)
        ph.empty()


def _record(feature: str, ms: float, ok: bool) -> None:
    st.session_state[state.LAST_AI_TS]    = time.time()
    st.session_state[state.AI_CALL_COUNT] = (
        st.session_state.get(state.AI_CALL_COUNT, 0) + 1
    )
    log = st.session_state.get(state.AI_LOG, [])
    log.append({"time": time.strftime("%H:%M:%S"),
                "feature": feature, "ms": round(ms), "ok": ok})
    st.session_state[state.AI_LOG] = log[-30:]


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════════
def _classify(msg: str) -> str:
    m = msg.lower()
    if "429" in msg or "resource_exhausted" in m or "quota" in m:
        return "quota"
    if "503" in msg or "unavailable" in m:
        return "unavailable"
    if "timeout" in m or "deadline" in m:
        return "timeout"
    return "other"


def _quota_countdown() -> None:
    """Show a user-friendly countdown after a real quota hit. Never shows stack traces."""
    ph = st.empty()
    for t in range(QUOTA_WAIT, 0, -1):
        ph.warning(
            f"Rate limit reached (free tier: 5 req/min). "
            f"Resuming in **{t}s**..."
        )
        time.sleep(1)
    ph.empty()
    st.info("Ready. Click the button to try again.")


def _sanitize(msg: str) -> str:
    """Strip verbose API internals, return a short user-readable string."""
    for prefix in ["400 ", "Error code:", "google.api_core", "grpc"]:
        if prefix in msg:
            msg = msg[msg.find(prefix):]
            break
    return msg[:100]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE CALL
# ══════════════════════════════════════════════════════════════════════════════
def _raw(prompt: str, json_mode: bool = False, feature: str = "call") -> str:
    """
    Blocking Gemini call with smart retry classification.

    Retry strategy:
      quota (429)       -> NO retry. Quota exhaustion means waiting is the only fix.
                           Show countdown, return empty string.
      unavailable (503) -> Retry up to 2x with backoff (15s, 30s). Transient.
      timeout           -> Retry once after 5s.
      other             -> Retry once after 3s.

    Using structured JSON output (response_mime_type=application/json) when
    json_mode=True. This is more reliable than parsing free-form text:
    Gemini enforces valid JSON at the API level, so _parse_json rarely needs
    to do heavy lifting.
    """
    client = _require_client()
    _enforce_rate_limit()

    cfg           = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    t0            = time.time()
    retried_503   = 0
    retried_once  = False

    while True:
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=cfg
            )
            text = (resp.text.strip() if hasattr(resp, "text") and resp.text
                    else "".join(
                        p.text for p in resp.candidates[0].content.parts
                        if hasattr(p, "text")
                    ).strip())
            _record(feature, (time.time() - t0) * 1000, ok=True)
            return text

        except Exception as exc:
            msg      = str(exc)
            etype    = _classify(msg)
            elapsed  = (time.time() - t0) * 1000

            if etype == "quota":
                _record(feature, elapsed, ok=False)
                _quota_countdown()
                return ""

            elif etype == "unavailable":
                if retried_503 < len(BACKOFF_503):
                    wait = BACKOFF_503[retried_503]
                    retried_503 += 1
                    st.toast(f"Service unavailable - retrying in {wait}s", icon="")
                    time.sleep(wait)
                    continue
                _record(feature, elapsed, ok=False)
                st.warning("Service temporarily unavailable. Please try again shortly.")
                return ""

            else:
                if not retried_once:
                    retried_once = True
                    time.sleep(5 if etype == "timeout" else 3)
                    continue
                _record(feature, elapsed, ok=False)
                st.warning(f"Request failed. Please try again. ({_sanitize(msg)})")
                return ""


def _parse_json(raw: str) -> Any:
    """
    Robust JSON extraction from Gemini output.
    Handles markdown fences, leading/trailing text, arrays and objects.
    """
    if not raw:
        return None
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    for op, cl in [("[", "]"), ("{", "}")]:
        s = cleaned.find(op)
        e = cleaned.rfind(cl) + 1
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s:e])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING
# ══════════════════════════════════════════════════════════════════════════════
def stream_call(prompt: str, feature: str = "stream") -> Generator:
    """
    Streaming generator. Pass directly to st.write_stream().
    Handles rate limiting, errors, and partial failures gracefully.
    On any error, yields a human-readable message instead of raising.
    """
    client = _require_client()
    _enforce_rate_limit()
    cfg = types.GenerateContentConfig(response_mime_type="text/plain")
    t0  = time.time()
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL, contents=prompt, config=cfg
        ):
            if chunk.text:
                yield chunk.text
        _record(feature, (time.time() - t0) * 1000, ok=True)
    except Exception as exc:
        msg   = str(exc)
        etype = _classify(msg)
        _record(feature, (time.time() - t0) * 1000, ok=False)
        if etype == "quota":
            yield "\n\nRate limit reached. Please wait ~60s and try again."
        elif etype == "unavailable":
            yield "\n\nService temporarily unavailable. Please try again."
        else:
            yield f"\n\nSomething went wrong. Please try again."


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED AI FEATURES
# (doc_hash as first arg ensures cache invalidates when document changes)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=7200)
def get_points_and_concepts(doc_hash: str, context: str) -> dict:
    """
    CONSOLIDATED: key_points + concepts in ONE call (was 2 separate calls).
    Cached for 2 hours. Shared across all tabs.
    """
    prompt = (
        f"{prompts.SYSTEM_JSON}\n\n"
        "Return a single JSON object with EXACTLY these two fields:\n"
        '{"key_points": ["insight 1",...,"insight 7"],'
        '"concepts":[{"term":"name","definition":"one sentence"},... exactly 5]}\n\n'
        f"{context}"
    )
    raw  = _raw(prompt, json_mode=True, feature="points+concepts")
    data = _parse_json(raw) or {}
    return {
        "key_points": data.get("key_points", [])
                      if isinstance(data.get("key_points"), list) else [],
        "concepts":   _validate_concepts(data.get("concepts", [])),
    }


@st.cache_data(show_spinner=False, ttl=7200)
def get_section_summaries(doc_hash: str, text: str) -> list:
    """
    Cached per-section summarization for long documents.
    Shared across analyze, quiz, and flashcards tabs — computed at most once
    per document per 2-hour window regardless of which tab triggers it first.
    """
    chunks  = chunk_doc(text)
    results = []
    bar     = st.progress(0, text=f"Reading {len(chunks)} sections...")
    for i, chunk in enumerate(chunks):
        bar.progress(int((i + 1) / len(chunks) * 55),
                     text=f"Section {i+1} of {len(chunks)}...")
        raw = _raw(
            prompts.chunk_summary(chunk, i + 1, len(chunks)),
            feature=f"chunk_{i+1}",
        )
        results.append(raw or chunk[:500])
    bar.empty()
    return results


@st.cache_data(show_spinner=False, ttl=7200)
def get_quiz(doc_hash: str, context: str, difficulty: str) -> list:
    """Cached quiz. Same doc + difficulty = no re-call."""
    raw  = _raw(prompts.quiz(context, difficulty, 5),
                json_mode=True, feature=f"quiz:{difficulty}")
    return _validate_quiz(_parse_json(raw))


@st.cache_data(show_spinner=False, ttl=7200)
def get_flashcards(doc_hash: str, context: str, n: int) -> list:
    """Cached flashcards. Same doc + count = no re-call."""
    raw  = _raw(prompts.flashcards(context, n),
                json_mode=True, feature=f"flashcards:{n}")
    return _validate_flashcards(_parse_json(raw))


@st.cache_data(show_spinner=False, ttl=7200)
def get_deep_dive(doc_hash: str, term: str, context: str) -> str:
    """
    Lazy deep dive — only called when user explicitly clicks the button.
    Cached so repeated clicks on the same concept cost zero API calls.
    """
    return _raw(prompts.deep_dive(term, context),
                feature=f"deep_dive:{term[:20]}")


@st.cache_data(show_spinner=False, ttl=1800)
def get_follow_ups(qa_hash: str, question: str, answer_excerpt: str) -> list:
    """
    Cached follow-up suggestions. qa_hash derived from question+answer pair.
    30 min TTL — less durable than analysis since follow-ups are contextual.
    """
    raw  = _raw(prompts.follow_up(question, answer_excerpt),
                json_mode=True, feature="follow_ups")
    data = _parse_json(raw)
    return data if isinstance(data, list) else []


# ══════════════════════════════════════════════════════════════════════════════
#  NON-CACHED (per-request by nature)
# ══════════════════════════════════════════════════════════════════════════════
def answer_question(context: str, question: str) -> Generator:
    """Streaming chat answer. Returns generator for st.write_stream()."""
    return stream_call(prompts.chat(context, question), feature="chat")


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION  (lightweight, no Pydantic required)
# ══════════════════════════════════════════════════════════════════════════════
def _validate_quiz(data: Any) -> list:
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q    = item.get("question", "")
        opts = item.get("options", [])
        ans  = str(item.get("answer", "")).strip()
        ans  = ans[0].upper() if ans else "A"
        if q and len(opts) >= 2:
            out.append({
                "question":    q,
                "options":     (list(opts) + [""] * 4)[:4],
                "answer":      ans,
                "explanation": item.get("explanation", ""),
            })
    return out


def _validate_flashcards(data: Any) -> list:
    if not isinstance(data, list):
        return []
    return [{"front": str(d.get("front", "")), "back": str(d.get("back", ""))}
            for d in data if isinstance(d, dict)
            and d.get("front") and d.get("back")]


def _validate_concepts(data: Any) -> list:
    if not isinstance(data, list):
        return []
    return [{"term": str(d.get("term", "")),
             "definition": str(d.get("definition", ""))}
            for d in data if isinstance(d, dict)
            and d.get("term") and d.get("definition")]


# ══════════════════════════════════════════════════════════════════════════════
#  OBSERVABILITY PANEL
# ══════════════════════════════════════════════════════════════════════════════
def render_diagnostics() -> None:
    """Render AI diagnostics. Call inside a sidebar expander."""
    count = st.session_state.get(state.AI_CALL_COUNT, 0)
    log   = st.session_state.get(state.AI_LOG, [])
    last  = st.session_state.get(state.LAST_AI_TS, 0.0)
    since = time.time() - last if last else 9999
    ready_in = max(0, CALL_GAP - since)

    c1, c2 = st.columns(2)
    c1.metric("API calls", count)
    c2.metric("Ready in", f"{ready_in:.0f}s" if ready_in > 1 else "Now")

    if log:
        st.markdown("**Last 5 calls:**")
        for e in reversed(log[-5:]):
            icon = "" if e["ok"] else ""
            st.caption(f"{icon} {e['time']} | {e['feature']} | {e['ms']}ms")


# ══════════════════════════════════════════════════════════════════════════════
#  COVERAGE MODES + ADAPTIVE LIMITS
# ══════════════════════════════════════════════════════════════════════════════

COVERAGE_MODES = {
    "Light": {
        "questions":      8,
        "flashcards":     10,
        "q_per_section":  3,
        "fc_per_section": 4,
        "label":          "8 questions · 10 cards",
        "time_est":       "~30–60s",
        "quota":          "low",
        "quota_color":    "#3fb950",
    },
    "Standard": {
        "questions":      15,
        "flashcards":     20,
        "q_per_section":  4,
        "fc_per_section": 5,
        "label":          "15 questions · 20 cards",
        "time_est":       "~1–2 min",
        "quota":          "medium",
        "quota_color":    "#d97706",
    },
    "Deep": {
        "questions":      30,
        "flashcards":     40,
        "q_per_section":  5,
        "fc_per_section": 7,
        "label":          "30 questions · 40 cards",
        "time_est":       "~3–5 min",
        "quota":          "high",
        "quota_color":    "#f87171",
    },
    "Full Document": {
        "questions":      None,   # adaptive: q_per_section × all sections
        "flashcards":     None,
        "q_per_section":  5,
        "fc_per_section": 8,
        "label":          "All sections covered",
        "time_est":       "~5–10 min",
        "quota":          "very high",
        "quota_color":    "#f85149",
    },
}


def available_modes(doc_text: str) -> list:
    """
    Return coverage modes appropriate for this document's word count.
    Prevents users from selecting modes that would waste quota on tiny documents
    or break token limits on large ones.
    """
    words = len(doc_text.split())
    modes = ["Light", "Standard"]
    if words >= 1500:
        modes.append("Deep")
    if words >= 3000:
        modes.append("Full Document")
    return modes


def estimate_generation(doc_text: str, coverage_mode: str,
                        content_type: str = "quiz") -> dict:
    """
    Calculate realistic estimates before the user clicks Generate.
    Used to show time, API call count, and quota warnings in the UI.

    content_type: "quiz" or "flashcards"
    """
    from chunker_utils import chunk_doc, is_long

    mode     = COVERAGE_MODES[coverage_mode]
    long     = is_long(doc_text)
    sections = chunk_doc(doc_text) if long else [doc_text]
    n_sec    = len(sections)

    per_sec  = mode["q_per_section"] if content_type == "quiz" else mode["fc_per_section"]
    target   = mode["questions"]     if content_type == "quiz" else mode["flashcards"]

    if coverage_mode == "Full Document":
        total_items    = n_sec * per_sec
        sections_used  = n_sec
    elif long:
        sections_used  = min(n_sec, max(1, (target or 15) // per_sec))
        total_items    = min(target or 15, sections_used * per_sec)
    else:
        sections_used  = 1
        total_items    = target or 15

    # Each section needs 1 API call; summaries may need n_sec calls on first run
    api_calls   = sections_used + (n_sec if long else 0)
    est_seconds = api_calls * 15    # conservative: 15s per call on free tier

    return {
        "total_items":    total_items,
        "sections_used":  sections_used,
        "total_sections": n_sec,
        "api_calls":      api_calls,
        "est_seconds":    est_seconds,
        "est_label":      _seconds_to_label(est_seconds),
        "is_large":       api_calls >= 6,
        "quota_color":    mode["quota_color"],
        "quota_label":    mode["quota"],
    }


def _seconds_to_label(s: int) -> str:
    if s < 60:
        return f"~{s}s"
    m = s // 60
    r = s % 60
    return f"~{m}m {r}s" if r else f"~{m} min"


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SECTION CACHED GENERATION
#  These are the atomic units of the chunked architecture.
#  Each section is cached independently so a failure in section 7 only
#  requires retrying section 7, not the entire document.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=7200)
def get_quiz_section(doc_hash: str, section_idx: int,
                     section_text: str, n: int, difficulty: str) -> list:
    """
    Generate quiz for one document section.
    Cache key: doc_hash + section_idx + n + difficulty.
    On cache hit: returns instantly, zero API calls.
    """
    raw = _raw(
        prompts.quiz_section(section_text, section_idx + 1, 1, n, difficulty),
        json_mode=True,
        feature=f"quiz_s{section_idx + 1}",
    )
    return _validate_quiz(_parse_json(raw))


@st.cache_data(show_spinner=False, ttl=7200)
def get_fc_section(doc_hash: str, section_idx: int,
                   section_text: str, n: int) -> list:
    """
    Generate flashcards for one document section.
    Cache key: doc_hash + section_idx + n.
    """
    raw = _raw(
        prompts.flashcards_section(section_text, section_idx + 1, 1, n),
        json_mode=True,
        feature=f"fc_s{section_idx + 1}",
    )
    return _validate_flashcards(_parse_json(raw))


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESSIVE GENERATION ORCHESTRATORS
#  These are NOT cached — they coordinate the per-section cached calls,
#  display progress UI, and assemble the final result.
# ══════════════════════════════════════════════════════════════════════════════

def generate_quiz(doc_hash: str, doc_text: str,
                  coverage_mode: str, difficulty: str) -> tuple:
    """
    Intelligent quiz generation that scales with document size and coverage mode.

    Short documents  → single consolidated call (fast, reliable)
    Long documents   → per-section calls with individual caching
                       (resilient: retry only failed sections)

    Returns: (questions: list, coverage_report: list)
    """
    from chunker_utils import is_long, chunk_doc, MAX_CHARS

    mode   = COVERAGE_MODES[coverage_mode]
    target = mode["questions"]          # None for Full Document
    q_per  = mode["q_per_section"]

    if not is_long(doc_text):
        # ── Short document: single call ───────────────────────────────────────
        cap = target or 15
        questions = get_quiz(doc_hash, doc_text[:MAX_CHARS], difficulty)
        questions = questions[:cap]
        report    = [{"section": 1, "q": len(questions), "covered": bool(questions)}]
        return questions, report

    # ── Long document: chunked generation ─────────────────────────────────────
    # Step 1: Ensure section summaries are ready
    with st.spinner("Loading document sections..."):
        sections = get_section_summaries(doc_hash, doc_text)

    n_sec = len(sections)
    total = target or (n_sec * q_per)   # Full Document: all sections

    # Determine how many sections to actually cover
    if coverage_mode == "Full Document":
        sections_to_cover = n_sec
    else:
        sections_to_cover = min(n_sec, max(1, (target or 15) // q_per + 1))

    all_questions = []
    coverage      = []

    bar = st.progress(0, text=f"Generating questions (0/{sections_to_cover} sections)...")

    for i in range(n_sec):
        # Stop once we have enough (non-Full Document modes)
        if target and len(all_questions) >= target:
            coverage.append({"section": i + 1, "q": 0, "covered": False,
                             "note": "target reached"})
            continue

        if i >= sections_to_cover and coverage_mode != "Full Document":
            coverage.append({"section": i + 1, "q": 0, "covered": False,
                             "note": "skipped"})
            continue

        pct  = int(((i + 1) / sections_to_cover) * 100)
        done = sum(1 for r in coverage if r.get("covered"))
        bar.progress(min(pct, 99),
                     text=f"Section {i + 1}/{sections_to_cover} "
                          f"({len(all_questions)} questions so far)...")

        chunk_q = get_quiz_section(doc_hash, i, sections[i], q_per, difficulty)
        all_questions.extend(chunk_q)
        coverage.append({
            "section": i + 1,
            "q":       len(chunk_q),
            "covered": bool(chunk_q),
        })

    bar.progress(100, text="Complete!")
    time.sleep(0.25)
    bar.empty()

    final = all_questions[:target] if target else all_questions
    return final, coverage


def generate_flashcards(doc_hash: str, doc_text: str,
                        coverage_mode: str) -> tuple:
    """
    Intelligent flashcard generation that scales with document size and coverage mode.
    Returns: (flashcards: list, coverage_report: list)
    """
    from chunker_utils import is_long, chunk_doc, MAX_CHARS

    mode      = COVERAGE_MODES[coverage_mode]
    target    = mode["flashcards"]      # None for Full Document
    fc_per    = mode["fc_per_section"]

    if not is_long(doc_text):
        cap   = target or 20
        cards = get_flashcards(doc_hash, doc_text[:MAX_CHARS], cap)
        report = [{"section": 1, "fc": len(cards), "covered": bool(cards)}]
        return cards, report

    with st.spinner("Loading document sections..."):
        sections = get_section_summaries(doc_hash, doc_text)

    n_sec  = len(sections)

    if coverage_mode == "Full Document":
        sections_to_cover = n_sec
    else:
        sections_to_cover = min(n_sec, max(1, (target or 20) // fc_per + 1))

    all_cards = []
    coverage  = []
    bar = st.progress(0, text=f"Creating flashcards (0/{sections_to_cover} sections)...")

    for i in range(n_sec):
        if target and len(all_cards) >= target:
            coverage.append({"section": i + 1, "fc": 0, "covered": False,
                             "note": "target reached"})
            continue

        if i >= sections_to_cover and coverage_mode != "Full Document":
            coverage.append({"section": i + 1, "fc": 0, "covered": False,
                             "note": "skipped"})
            continue

        pct = int(((i + 1) / sections_to_cover) * 100)
        bar.progress(min(pct, 99),
                     text=f"Section {i + 1}/{sections_to_cover} "
                          f"({len(all_cards)} cards so far)...")

        chunk_fc = get_fc_section(doc_hash, i, sections[i], fc_per)
        all_cards.extend(chunk_fc)
        coverage.append({
            "section": i + 1,
            "fc":      len(chunk_fc),
            "covered": bool(chunk_fc),
        })

    bar.progress(100, text="Complete!")
    time.sleep(0.25)
    bar.empty()

    final = all_cards[:target] if target else all_cards
    return final, coverage