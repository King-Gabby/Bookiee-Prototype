"""
Bookiee AI — MVP v1.1
Week 1+2 upgrade: caching · prompts registry · smart chunking ·
retry + rate-limit handling · structured JSON output · refresh bug fixed
"""

import streamlit as st
from google import genai
from google.genai import types
import json, io, re, time, hashlib
from datetime import datetime
from typing import Any, List
import pdfplumber
from pydantic import BaseModel, ValidationError, field_validator
import prompts

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MODEL         = "gemini-2.5-flash"
CHUNK_WORDS   = 1600   # ~2 000 tokens per chunk
OVERLAP_WORDS = 160    # overlap words kept at chunk boundaries
MAX_CHARS     = 8000   # single-call context budget
APP_VERSION   = "1.1.0"

DEMO_TEXT = (
    "The Science of Deep Work: How Focus Became the New Superpower\n\n"
    "In an age of infinite distraction, the ability to concentrate deeply on cognitively demanding "
    "tasks has become one of the rarest and most valuable skills in the modern economy. Cal Newport, "
    "a computer science professor at Georgetown University, coined the term deep work to describe "
    "professional activities performed in a state of distraction-free concentration that push your "
    "cognitive capabilities to their limit.\n\n"
    "The Neurological Case for Focus\n"
    "When you spend time in a state of deep work, two things happen at the cellular level. First, "
    "myelin - a white tissue that develops around neurons - grows thicker with use, making those "
    "neural pathways fire faster and cleaner. Second, the default mode network, associated with "
    "mind-wandering and self-referential thought, quiets down. This allows the prefrontal cortex, "
    "responsible for executive function and focused attention, to operate at full capacity.\n\n"
    "Shallow Work: The Silent Productivity Killer\n"
    "Contrast deep work with shallow work - non-cognitively demanding tasks often performed while "
    "distracted. Answering emails, attending status meetings, and scrolling through notifications "
    "all qualify. Studies from UC Irvine found it takes an average of 23 minutes to fully regain "
    "concentration after an interruption.\n\n"
    "The Four Philosophies of Depth\n"
    "Newport identifies four scheduling philosophies: The Monastic (eliminate all shallow "
    "obligations), The Bimodal (alternate deep and shallow seasons), The Rhythmic (daily deep work "
    "ritual), and The Journalistic (fit deep work wherever you can).\n\n"
    "Training the Attention Muscle\n"
    "Deep work is a skill, not a switch. Your capacity for concentration is like a muscle: it "
    "atrophies without use and strengthens with training. Boredom is the gym.\n\n"
    "The Economic Argument\n"
    "In the new economy, two groups thrive: those who can master hard things quickly, and those "
    "who can produce at an elite level. Both depend on deep work. In a world where attention is "
    "for sale, those who own theirs have a decisive competitive advantage."
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bookiee AI",
    page_icon="\U0001f4da",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.bk-header{text-align:center;padding:1.5rem 0 .75rem}
.bk-header h1{font-family:'Playfair Display',serif;font-size:2.6rem;
  background:linear-gradient(135deg,#2ea043,#3fb950,#58d168);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem}
.bk-header p{color:#8b949e;font-size:.9rem;margin:0}
.bk-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem}
.bk-card-title{font-size:.7rem;font-weight:700;color:#3fb950;text-transform:uppercase;letter-spacing:.9px;margin-bottom:.75rem}
.bk-card-body{font-size:.875rem;color:#c9d1d9;line-height:1.75}
.bk-pill{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:20px;padding:5px 13px;font-size:.78rem;color:#c9d1d9;margin:3px;line-height:1.4}
.bk-concept{border-left:2.5px solid #3fb950;padding:10px 14px;margin-bottom:10px;background:#0d1117;border-radius:0 8px 8px 0}
.bk-concept-term{color:#e6edf3;font-size:.875rem;font-weight:600;margin-bottom:3px}
.bk-concept-def{color:#8b949e;font-size:.8rem;line-height:1.55}
.demo-banner{background:rgba(46,160,67,.08);border:1px solid rgba(46,160,67,.2);border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:.8rem;color:#8b949e;display:flex;align-items:center;gap:8px}
.demo-dot{width:7px;height:7px;background:#3fb950;border-radius:50%;flex-shrink:0;animation:bkpulse 1.5s infinite}
@keyframes bkpulse{0%,100%{opacity:.4}50%{opacity:1}}
.doc-meta{display:flex;gap:16px;flex-wrap:wrap;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:.78rem;color:#8b949e}
.doc-meta span strong{color:#c9d1d9}
.chunk-badge{background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.25);border-radius:20px;padding:4px 12px;font-size:.72rem;color:#d97706;margin-bottom:12px;display:inline-block}
.score-badge{display:inline-block;background:rgba(46,160,67,.12);border:1px solid rgba(46,160,67,.3);border-radius:20px;padding:5px 14px;font-size:.8rem;color:#3fb950;font-weight:600;margin-bottom:14px}
.rate-warn{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:8px;padding:10px 14px;font-size:.8rem;color:#f87171;margin-bottom:10px}
.fc{background:#0d1117;border:1.5px solid #30363d;border-radius:10px;padding:1.2rem;text-align:center;min-height:110px;display:flex;flex-direction:column;align-items:center;justify-content:center;transition:border-color .2s}
.fc.flipped{border-color:#3fb950;background:#070f0a}
.fc-label{font-size:.65rem;color:#484f57;text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px}
.fc-front{font-size:.9rem;font-weight:600;color:#e6edf3;line-height:1.45}
.fc-back{font-size:.8rem;color:#7ee787;line-height:1.6}
.empty-state{text-align:center;padding:3rem 1rem;color:#484f57}
.empty-state .icon{font-size:3rem;margin-bottom:1rem}
.stButton>button{background:linear-gradient(135deg,#2ea043,#3fb950)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:.5rem 1.5rem!important;transition:all .2s!important}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important}
.stButton>button:disabled{opacity:.35!important;transform:none!important}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d}
.stTextArea textarea,.stTextInput input{background:#0d1117!important;border-color:#30363d!important;color:#e6edf3!important}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC VALIDATION MODELS
# ══════════════════════════════════════════════════════════════════════════════
class QuizQuestion(BaseModel):
    question:    str
    options:     List[str]
    answer:      str
    explanation: str

    @field_validator("answer")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip()[0].upper() if v.strip() else "A"

    @field_validator("options")
    @classmethod
    def four_opts(cls, v: List[str]) -> List[str]:
        return (v + [""] * 4)[:4]


class Flashcard(BaseModel):
    front: str
    back:  str


class Concept(BaseModel):
    term:       str
    definition: str


def _validate(model, data: Any) -> list:
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        try:
            out.append(model(**item).model_dump())
        except (ValidationError, TypeError):
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  AI ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_client():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        st.error("Add GEMINI_API_KEY in Settings -> Secrets.")
        st.stop()
    return genai.Client(api_key=key)


def _raw_call(prompt: str, system: str, json_mode: bool = False, retries: int = 3) -> str:
    """Blocking call with retry + rate-limit back-off."""
    client = get_client()
    cfg    = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    full = f"{system}\n\n{prompt}"
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL, contents=full, config=cfg)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            return "".join(
                p.text for p in resp.candidates[0].content.parts if hasattr(p, "text")
            ).strip()
        except Exception as exc:
            msg   = str(exc)
            is_rl = any(x in msg for x in ["429", "rate", "quota", "exhausted"])
            last  = attempt == retries - 1
            if is_rl and not last:
                wait = (attempt + 1) * 12
                st.markdown(f'<div class="rate-warn">Rate limit - retrying in {wait}s ({attempt+1}/{retries})</div>',
                            unsafe_allow_html=True)
                time.sleep(wait)
            elif last:
                st.error(f"AI call failed after {retries} attempts: {msg}")
                return ""
            else:
                time.sleep(2)
    return ""


def _parse_json(raw: str) -> Any:
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


@st.cache_data(show_spinner=False, ttl=7200)
def cached_json(cache_key: str, prompt: str) -> Any:
    """
    Cached JSON call. cache_key = doc_hash + task.
    ttl=7200 -> results survive 2h of reruns without re-hitting the API.
    """
    raw = _raw_call(prompt, system=prompts.SYSTEM_JSON, json_mode=True)
    return _parse_json(raw) if raw else None


def stream_gemini(prompt: str):
    """Streaming generator - pass to st.write_stream."""
    client = get_client()
    cfg    = types.GenerateContentConfig(response_mime_type="text/plain")
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=f"{prompts.SYSTEM_STUDY}\n\n{prompt}",
            config=cfg,
        ):
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        msg = str(exc)
        yield ("\n\nRate limit - wait 60s and retry." if "429" in msg or "quota" in msg
               else f"\n\nError: {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT PROCESSING + SMART CHUNKING  (Week 2)
# ══════════════════════════════════════════════════════════════════════════════
def extract_text(f) -> str:
    if f.name.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(f.read())) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages[:50])
    return f.read().decode("utf-8", errors="ignore")


def make_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def doc_meta(text: str) -> dict:
    w = len(text.split())
    return {"words": w, "chars": len(text), "read_mins": max(1, round(w / 238))}


def chunk_doc(text: str) -> list:
    """Sentence-aware chunking with overlap for context continuity."""
    sentences       = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur, wc = [], [], 0
    for sent in sentences:
        sw = len(sent.split())
        if wc + sw > CHUNK_WORDS and cur:
            chunks.append(" ".join(cur))
            overlap, carried = [], 0
            for s in reversed(cur):
                n = len(s.split())
                if carried + n <= OVERLAP_WORDS:
                    overlap.insert(0, s); carried += n
                else:
                    break
            cur, wc = overlap, carried
        cur.append(sent); wc += sw
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def is_long(text: str) -> bool:
    return len(text) > MAX_CHARS * 1.1


@st.cache_data(show_spinner=False, ttl=7200)
def get_section_summaries(doc_hash: str, text: str) -> list:
    """
    Cached per-section summarization.
    Shared across tabs: analyze, quiz, flashcards all reuse the same chunk work.
    """
    chunks = chunk_doc(text)
    out    = []
    bar    = st.progress(0, text=f"Reading {len(chunks)} sections...")
    for i, chunk in enumerate(chunks):
        bar.progress(int((i + 1) / len(chunks) * 55), text=f"Section {i+1}/{len(chunks)}...")
        raw = _raw_call(prompts.chunk_summary(chunk, i + 1, len(chunks)),
                        system=prompts.SYSTEM_STUDY)
        out.append(raw or chunk[:500])
    bar.empty()
    return out


def smart_chat_ctx(text: str, question: str) -> str:
    """
    Keyword-match each chunk to the question, return best + neighbours.
    No embeddings needed at this scale - keyword overlap is highly effective
    for study material where questions mirror document vocabulary directly.
    """
    if not is_long(text):
        return text[:MAX_CHARS]
    chunks  = chunk_doc(text)
    q_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    scored  = sorted(enumerate(chunks),
                     key=lambda ic: len(q_words & set(ic[1].lower().split())),
                     reverse=True)
    best = scored[0][0]
    return " ".join(chunks[max(0, best - 1): best + 2])[:MAX_CHARS]


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEF: dict = {
    "doc_text":          "",
    "is_demo":           False,
    "_doc_source_id":    "",   # THE REFRESH BUG FIX: stable doc identity
    "analysis":          None,
    "chat_history":      [],
    "quiz_data":         [],
    "quiz_answers":      {},
    "fc_data":           [],
    "fc_flipped":        set(),
    "simplify_text":     "",
    "simplify_audience": "",
    "docs_analyzed":     0,
    "questions_asked":   0,
    "quizzes_taken":     0,
    "session_start":     datetime.now().strftime("%H:%M"),
    "_pending_q":        "",
}
for k, v in _DEF.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_derived() -> None:
    for k in ["analysis","chat_history","quiz_data","quiz_answers",
              "fc_data","simplify_text","simplify_audience"]:
        st.session_state[k] = _DEF[k]
    st.session_state.fc_flipped = set()


def set_document(text: str, source_id: str, is_demo: bool = False) -> None:
    """
    Only resets derived state when the document ACTUALLY changes.

    Root cause of the refresh bug:
      Streamlit reruns on every widget interaction.
      The old code called reset_doc_state() every time st.file_uploader()
      returned a file - which it does on EVERY rerun, not just on upload.
      So quiz answers, analysis, and flashcards were wiped on every button click.

    The fix: compare source_id. If it matches the stored _doc_source_id,
    the document hasn't changed - skip the reset entirely.
    """
    if source_id == st.session_state._doc_source_id:
        return
    st.session_state.doc_text       = text
    st.session_state._doc_source_id = source_id
    st.session_state.is_demo        = is_demo
    _reset_derived()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### \U0001f4da Bookiee AI")
    st.caption(f"v{APP_VERSION}  Gemini 2.5 Flash (free)")
    st.divider()

    st.markdown("**Try the demo**")
    if st.button("Load Demo Article", use_container_width=True):
        set_document(DEMO_TEXT, source_id="__demo__", is_demo=True)
        st.toast("Demo loaded!", icon="\U0001f4d6")
        st.rerun()

    st.markdown("**Upload a file**")
    uploaded = st.file_uploader("PDF / TXT / MD", type=["txt","pdf","md"],
                                label_visibility="collapsed")
    if uploaded:
        file_id = f"file::{uploaded.name}::{uploaded.size}"
        if file_id != st.session_state._doc_source_id:
            with st.spinner("Extracting..."):
                text = extract_text(uploaded)
            if text.strip():
                set_document(text, source_id=file_id)
                st.toast(f"{uploaded.name} loaded!", icon="\U0001f4c4")
            else:
                st.error("Could not extract text. Try pasting directly.")

    st.markdown("**Or paste text**")
    pasted = st.text_area("", height=140,
                          placeholder="Paste article, notes, chapter...",
                          label_visibility="collapsed", key="paste_area")
    if pasted:
        paste_id = f"paste::{hashlib.md5(pasted.encode()).hexdigest()[:10]}"
        if paste_id != st.session_state._doc_source_id:
            set_document(pasted, source_id=paste_id)

    has_doc  = len(st.session_state.doc_text.strip()) > 80
    d_hash   = make_hash(st.session_state.doc_text) if has_doc else ""
    long_doc = is_long(st.session_state.doc_text)   if has_doc else False

    if has_doc:
        meta = doc_meta(st.session_state.doc_text)
        st.markdown(
            f"<div class='doc-meta'>"
            f"<span>\U0001f4dd <strong>{meta['words']:,}</strong> words</span>"
            f"<span>\u23f1 <strong>~{meta['read_mins']} min</strong></span>"
            f"<span>\U0001f4be <strong>{meta['chars']:,}</strong> chars</span>"
            f"</div>", unsafe_allow_html=True)
        if long_doc:
            nc = len(chunk_doc(st.session_state.doc_text))
            st.markdown(f'<div class="chunk-badge">Long doc - {nc} sections - smart chunking on</div>',
                        unsafe_allow_html=True)
        if st.button("Clear everything", use_container_width=True):
            st.session_state.doc_text       = ""
            st.session_state._doc_source_id = ""
            st.session_state.is_demo        = False
            _reset_derived()
            st.rerun()
    else:
        st.caption("Load demo or add a document to begin.")

    st.divider()
    st.markdown("**Session**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Docs",    st.session_state.docs_analyzed)
    c2.metric("Q&As",    st.session_state.questions_asked)
    c3.metric("Quizzes", st.session_state.quizzes_taken)
    st.caption(f"Started {st.session_state.session_start}")

    st.divider()
    with st.expander("How it works"):
        st.markdown("""
1. **Upload** PDF / .txt / .md or paste text
2. **Analyze** - summary, key points, concepts
3. **Chat** - ask anything; long docs use smart section matching
4. **Study** - quiz, flashcards, simplified explanation

Results are **cached** - no redundant API calls.
Long documents are **chunked** and synthesized automatically.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bk-header">
  <h1>\U0001f4da Bookiee AI</h1>
  <p>Upload any document - understand it deeply, study it fast.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "\U0001f4c4  Analyze Document",
    "\U0001f4ac  Ask Questions",
    "\U0001f9e0  Study Mode",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 - ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not has_doc:
        st.markdown("<div class='empty-state'><div class='icon'>\U0001f4c2</div>"
                    "<div>Load the demo or upload a document to begin.</div></div>",
                    unsafe_allow_html=True)
    else:
        if st.session_state.is_demo:
            st.markdown("<div class='demo-banner'><div class='demo-dot'></div>"
                        "<span>Demo mode - <strong>Deep Work</strong> by Cal Newport. "
                        "Upload your own from the sidebar.</span></div>",
                        unsafe_allow_html=True)

        if st.session_state.analysis is None:
            if st.button("Analyze Document", use_container_width=True):
                prog  = st.progress(0, "Starting...")
                sum_ph = st.empty()

                if long_doc:
                    prog.progress(5, "Chunking document...")
                    sections = get_section_summaries(d_hash, st.session_state.doc_text)
                    prog.progress(60, "Synthesizing summary...")
                    full_summary = ""
                    for chunk in stream_gemini(prompts.synthesize_summary(sections)):
                        full_summary += chunk
                        sum_ph.markdown(
                            f'<div class="bk-card"><div class="bk-card-title">Summary</div>'
                            f'<div class="bk-card-body">{full_summary}</div></div>',
                            unsafe_allow_html=True)
                    sum_ph.empty()
                    prog.progress(78, "Key points...")
                    kp_raw   = cached_json(d_hash+":kp",   prompts.synthesize_key_points(sections))
                    prog.progress(90, "Concepts...")
                    conc_raw = cached_json(d_hash+":conc", prompts.synthesize_concepts(sections))
                else:
                    ctx = st.session_state.doc_text[:MAX_CHARS]
                    prog.progress(10, "Generating summary...")
                    full_summary = ""
                    for chunk in stream_gemini(prompts.summary(ctx)):
                        full_summary += chunk
                        sum_ph.markdown(
                            f'<div class="bk-card"><div class="bk-card-title">Summary</div>'
                            f'<div class="bk-card-body">{full_summary}</div></div>',
                            unsafe_allow_html=True)
                    sum_ph.empty()
                    prog.progress(60, "Key points...")
                    kp_raw   = cached_json(d_hash+":kp",   prompts.key_points(ctx))
                    prog.progress(85, "Concepts...")
                    conc_raw = cached_json(d_hash+":conc", prompts.concepts(ctx))

                prog.progress(100, "Done!")
                time.sleep(0.2)
                prog.empty()
                st.session_state.analysis = {
                    "summary":    full_summary,
                    "key_points": kp_raw if isinstance(kp_raw, list) else [],
                    "concepts":   _validate(Concept, conc_raw),
                }
                st.session_state.docs_analyzed += 1
                st.toast("Analysis complete!", icon="\u2705")
                st.rerun()

        if st.session_state.analysis:
            a = st.session_state.analysis
            col_r, col_e = st.columns(2)
            with col_r:
                if st.button("Re-analyze", use_container_width=True):
                    st.session_state.analysis = None
                    st.rerun()
            with col_e:
                md = ("# Bookiee AI - Analysis\n\n## Summary\n" + a["summary"] +
                      "\n\n## Key Points\n" + "\n".join(f"- {p}" for p in a["key_points"]) +
                      "\n\n## Key Concepts\n" +
                      "".join(f"**{c.get('term','')}** - {c.get('definition','')}\n\n"
                              for c in a["concepts"]))
                st.download_button("Export Markdown", data=md,
                                   file_name="bookiee_analysis.md",
                                   mime="text/markdown", use_container_width=True)
            if long_doc:
                nc = len(chunk_doc(st.session_state.doc_text))
                st.markdown(f'<div class="chunk-badge">Full-document analysis across {nc} sections</div>',
                            unsafe_allow_html=True)
            st.divider()
            st.markdown(f'<div class="bk-card"><div class="bk-card-title">Summary</div>'
                        f'<div class="bk-card-body">{a["summary"]}</div></div>',
                        unsafe_allow_html=True)
            pills = " ".join(f'<span class="bk-pill"> {p}</span>' for p in a["key_points"])
            st.markdown(f'<div class="bk-card"><div class="bk-card-title">Key Points</div>'
                        f'{pills}</div>', unsafe_allow_html=True)
            st.markdown('<div class="bk-card"><div class="bk-card-title">Key Concepts</div>',
                        unsafe_allow_html=True)
            for c in a["concepts"]:
                term, defi = c.get("term",""), c.get("definition","")
                st.markdown(f'<div class="bk-concept"><div class="bk-concept-term">{term}</div>'
                            f'<div class="bk-concept-def">{defi}</div></div>',
                            unsafe_allow_html=True)
                with st.expander(f"Deep dive: {term}"):
                    dive_ctx = (smart_chat_ctx(st.session_state.doc_text, term)
                                if long_doc else st.session_state.doc_text[:MAX_CHARS])
                    st.write_stream(stream_gemini(prompts.deep_dive(term, dive_ctx)))
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 - CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not has_doc:
        st.markdown("<div class='empty-state'><div class='icon'>\U0001f4ac</div>"
                    "<div>Add a document first, then ask anything about it.</div></div>",
                    unsafe_allow_html=True)
    else:
        if long_doc:
            st.markdown('<div class="chunk-badge">Long doc - smart section matching active</div>',
                        unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown("**Quick questions:**")
            qa, qb, qc = st.columns(3)
            for col, q in zip([qa, qb, qc], [
                "What is the main argument?",
                "Explain the key concept simply",
                "Give me 3 actionable takeaways",
            ]):
                if col.button(q, use_container_width=True, key=f"qs_{q[:8]}"):
                    st.session_state._pending_q = q
                    st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="\U0001f9d1" if msg["role"]=="user" else "\U0001f4da"):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask anything about your document...")
        if st.session_state._pending_q:
            user_input, st.session_state._pending_q = st.session_state._pending_q, ""

        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            st.session_state.questions_asked += 1
            ctx = smart_chat_ctx(st.session_state.doc_text, user_input)
            with st.chat_message("assistant", avatar="\U0001f4da"):
                answer = st.write_stream(stream_gemini(prompts.chat(ctx, user_input)))
            st.session_state.chat_history.append({"role":"assistant","content":answer})

            suggestions = cached_json(
                d_hash + f":fu:{len(st.session_state.chat_history)}",
                prompts.follow_up(user_input, str(answer)),
            )
            if suggestions and isinstance(suggestions, list):
                st.markdown("**You might also ask:**")
                s_cols = st.columns(min(len(suggestions), 3))
                for i, (col, sug) in enumerate(zip(s_cols, suggestions[:3])):
                    if col.button(sug, key=f"sug_{i}_{len(st.session_state.chat_history)}",
                                  use_container_width=True):
                        st.session_state._pending_q = sug
                        st.rerun()
            # No st.rerun() here - avoids unnecessary full page cycle

        if st.session_state.chat_history:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 - STUDY MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not has_doc:
        st.markdown("<div class='empty-state'><div class='icon'>\U0001f9e0</div>"
                    "<div>Add a document to generate study materials.</div></div>",
                    unsafe_allow_html=True)
    else:
        mode = st.radio("", ["\U0001f4dd  Quiz","\U0001f0cf  Flashcards","\u270f\ufe0f  Simplify"],
                        horizontal=True, label_visibility="collapsed")

        # -- QUIZ --
        if mode == "\U0001f4dd  Quiz":
            diff = st.select_slider("Difficulty", ["Easy","Medium","Hard"], value="Medium")
            if st.button("Generate Quiz", use_container_width=True):
                with st.spinner("Building quiz..."):
                    if long_doc:
                        sections = get_section_summaries(d_hash, st.session_state.doc_text)
                        raw = cached_json(d_hash+f":quiz:{diff}",
                                          prompts.synthesize_quiz(sections, diff, 5))
                    else:
                        raw = cached_json(d_hash+f":quiz:{diff}",
                                          prompts.quiz(st.session_state.doc_text[:MAX_CHARS], diff, 5))
                    validated = _validate(QuizQuestion, raw)
                    if validated:
                        st.session_state.quiz_data    = validated
                        st.session_state.quiz_answers = {}
                        st.session_state.quizzes_taken += 1
                        st.toast("Quiz ready!", icon="\U0001f4dd")
                    else:
                        st.error("Quiz generation failed - try again.")

            if st.session_state.quiz_data:
                total    = len(st.session_state.quiz_data)
                answered = len(st.session_state.quiz_answers)
                correct  = sum(1 for i,q in enumerate(st.session_state.quiz_data)
                               if st.session_state.quiz_answers.get(i)==q.get("answer",""))
                st.markdown(f'<div class="score-badge">{correct}/{total} correct - {answered} answered</div>',
                            unsafe_allow_html=True)
                if answered == total > 0:
                    rmd = f"# Quiz Results\nScore: {correct}/{total}\n\n"
                    for i,q in enumerate(st.session_state.quiz_data):
                        g = st.session_state.quiz_answers.get(i,"?")
                        rmd += (f"**Q{i+1}.** {q.get('question','')}\n"
                                f"Yours: {g} | Correct: {q.get('answer','')}\n"
                                f"Explanation: {q.get('explanation','')}\n\n")
                    st.download_button("Export Results", data=rmd,
                                       file_name="bookiee_quiz.md", mime="text/markdown")
                for i, q in enumerate(st.session_state.quiz_data):
                    done  = i in st.session_state.quiz_answers
                    given = st.session_state.quiz_answers.get(i,"")
                    ans   = q.get("answer","")
                    icon  = "\u2705" if done and given==ans else ("\u274c" if done else "\u25cb")
                    with st.expander(f"{icon}  Q{i+1}. {q.get('question','')[:80]}", expanded=not done):
                        chosen = st.radio("", q.get("options",[]),
                                          key=f"qr_{i}_{d_hash}", label_visibility="collapsed")
                        if not done:
                            if st.button("Submit", key=f"qs_{i}_{d_hash}"):
                                st.session_state.quiz_answers[i] = (chosen or " ")[0]
                                st.rerun()
                        else:
                            if given == ans:
                                st.success(f"Correct! Answer: **{ans}**")
                            else:
                                st.error(f"You chose **{given}** - Correct: **{ans}**")
                            st.caption(q.get("explanation",""))

        # -- FLASHCARDS --
        elif mode == "\U0001f0cf  Flashcards":
            n = st.slider("Number of cards", 4, 16, 8, step=2)
            if st.button("Generate Flashcards", use_container_width=True):
                with st.spinner(f"Creating {n} flashcards..."):
                    if long_doc:
                        sections = get_section_summaries(d_hash, st.session_state.doc_text)
                        raw = cached_json(d_hash+f":fc:{n}",
                                          prompts.synthesize_flashcards(sections, n))
                    else:
                        raw = cached_json(d_hash+f":fc:{n}",
                                          prompts.flashcards(st.session_state.doc_text[:MAX_CHARS], n))
                    validated = _validate(Flashcard, raw)
                    if validated:
                        st.session_state.fc_data    = validated
                        st.session_state.fc_flipped = set()
                        st.toast(f"{len(validated)} flashcards ready!", icon="\U0001f0cf")
                    else:
                        st.error("Flashcard generation failed - try again.")

            if st.session_state.fc_data:
                total_fc = len(st.session_state.fc_data)
                revealed = len(st.session_state.fc_flipped)
                st.markdown(f'<div class="score-badge">{revealed}/{total_fc} revealed</div>',
                            unsafe_allow_html=True)
                fa, fb = st.columns(2)
                if fa.button("Flip all",  use_container_width=True):
                    st.session_state.fc_flipped = set(range(total_fc)); st.rerun()
                if fb.button("Reset all", use_container_width=True):
                    st.session_state.fc_flipped = set();                st.rerun()
                cols = st.columns(2)
                for i, card in enumerate(st.session_state.fc_data):
                    flipped = i in st.session_state.fc_flipped
                    content = card.get("back" if flipped else "front","")
                    with cols[i % 2]:
                        st.markdown(
                            f'<div class="{"fc flipped" if flipped else "fc"}">'
                            f'<div class="fc-label">{"BACK" if flipped else "FRONT"}</div>'
                            f'<div class="{"fc-back" if flipped else "fc-front"}">{content}</div></div>',
                            unsafe_allow_html=True)
                        if st.button("Flip", key=f"fc_{i}_{d_hash}", use_container_width=True):
                            if flipped: st.session_state.fc_flipped.discard(i)
                            else:       st.session_state.fc_flipped.add(i)
                            st.rerun()

        # -- SIMPLIFY --
        elif mode == "\u270f\ufe0f  Simplify":
            audience = st.radio("Explain as a:",
                                ["Complete beginner","Curious teenager","Busy professional"],
                                horizontal=True)
            if st.button("Simplify", use_container_width=True):
                ctx    = (" ".join(chunk_doc(st.session_state.doc_text)[:2])[:MAX_CHARS]
                          if long_doc else st.session_state.doc_text[:MAX_CHARS])
                holder = st.empty()
                full   = ""
                for chunk in stream_gemini(prompts.simplify(ctx, audience)):
                    full += chunk
                    holder.markdown(f'<div class="bk-card"><div class="bk-card-title">For: {audience}</div>'
                                    f'<div class="bk-card-body">{full}</div></div>',
                                    unsafe_allow_html=True)
                holder.empty()
                st.session_state.simplify_text     = full
                st.session_state.simplify_audience = audience
                # No st.rerun() - renders immediately below

            if st.session_state.simplify_text:
                st.markdown(
                    f'<div class="bk-card"><div class="bk-card-title">'
                    f'{st.session_state.simplify_audience}</div>'
                    f'<div class="bk-card-body">{st.session_state.simplify_text}</div></div>',
                    unsafe_allow_html=True)
                st.download_button("Export Explanation",
                                   data=st.session_state.simplify_text,
                                   file_name="bookiee_simplified.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    f"<p style='text-align:center;font-size:.75rem;color:#484f57'>"
    f"Bookiee AI v{APP_VERSION} - Streamlit + Gemini 2.5 Flash - "
    "<a href='https://github.com' style='color:#3fb950;text-decoration:none'>GitHub</a></p>",
    unsafe_allow_html=True)