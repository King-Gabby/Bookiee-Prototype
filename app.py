"""
Bookiee AI — Full MVP
Upload → Analyze → Study Smarter
Powered by Google Gemini 2.5 Flash (free tier)
"""

import streamlit as st
from google import genai
import json, io, time
from datetime import datetime
import pdfplumber

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MODEL       = "gemini-2.5-flash"
MAX_CHARS   = 8000
APP_VERSION = "1.0.0"

DEMO_TEXT = """
The Science of Deep Work: How Focus Became the New Superpower

In an age of infinite distraction, the ability to concentrate deeply on cognitively demanding
tasks has become one of the rarest and most valuable skills in the modern economy. I made a research some time ago, Cal Newport,
a computer science professor at Georgetown University (USA), coined the term "deep work" to describe
professional activities performed in a state of distraction-free concentration that push your
cognitive capabilities to their limit.

The Neurological Case for Focus
When you spend time in a state of deep work, two things happen at the cellular level. First,
myelin, a white tissue that develops around neurons, grows thicker with use, making those
neural pathways fire faster and cleaner. Second, the default mode network, associated with
mind-wandering and self-referential thought, quiets down. This allows the prefrontal cortex,
responsible for executive function and focused attention, to operate at full capacity.
The result is not just better work, it is structurally better thinking.

Shallow Work: The Silent Productivity Killer
Contrast deep work with what Newport calls "shallow work", non-cognitively demanding,
logistical-style tasks often performed while distracted. Answering emails, attending status
meetings, and scrolling through notifications all qualify. The insidious problem is that
shallow work creates the illusion of productivity. You feel busy, yet produce little of
lasting value. Studies from the University of California Irvine found that it takes an
average of 23 minutes to fully regain concentration after an interruption.

The Four Philosophies of Depth
Newport identifies four scheduling philosophies for integrating deep work:
1. The Monastic Philosophy :— eliminate all shallow obligations entirely, like writers who
   disappear for months.
2. The Bimodal Philosophy :— alternate between deep and shallow seasons, dedicating clear
   stretches to depth.
3. The Rhythmic Philosophy :— build a consistent daily ritual of deep work, the most practical
   for most people.
4. The Journalistic Philosophy :— fit deep work wherever you can, like a journalist filing
   stories on deadline.

Training the Attention Muscle
Deep work is a skill, and not just a switch. Newport recommends starting with 1-hour blocks and
building toward 4-hour sessions. The key insight is that your capacity for concentration
is like a muscle: it atrophies without use and strengthens with training. Boredom is the
gym, learning to resist the urge to reach for your phone in idle moments builds the same
neural tolerance for sustained focus that deep work demands.

The Economic Argument
In the new economy, two groups will thrive: those who can master hard things quickly, and
those who can produce at an elite level. Both outcomes depend on deep work. The ability to
learn rapidly and perform at the highest tier of intellectual output is directly proportional
to the quantity and quality of time spent in a state of deep, distraction-free concentration.
In a world where most people's attention is for sale, those who own theirs have a decisive
competitive advantage.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bookiee AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.bk-header { text-align:center; padding:1.5rem 0 .75rem; }
.bk-header h1 {
    font-family:'Playfair Display',serif; font-size:2.6rem;
    background:linear-gradient(135deg,#2ea043,#3fb950,#58d168);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.2rem;
}
.bk-header p { color:#8b949e; font-size:.9rem; margin:0; }

.bk-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:1rem;
}
.bk-card-title {
    font-size:.7rem; font-weight:700; color:#3fb950;
    text-transform:uppercase; letter-spacing:.9px; margin-bottom:.75rem;
}
.bk-card-body { font-size:.875rem; color:#c9d1d9; line-height:1.75; }

.bk-pill {
    display:inline-block; background:#21262d; border:1px solid #30363d;
    border-radius:20px; padding:5px 13px; font-size:.78rem;
    color:#c9d1d9; margin:3px; line-height:1.4;
}

.bk-concept {
    border-left:2.5px solid #3fb950; padding:10px 14px;
    margin-bottom:10px; background:#0d1117; border-radius:0 8px 8px 0;
}
.bk-concept-term { color:#e6edf3; font-size:.875rem; font-weight:600; margin-bottom:3px; }
.bk-concept-def  { color:#8b949e; font-size:.8rem; line-height:1.55; }

.demo-banner {
    background:rgba(46,160,67,.08); border:1px solid rgba(46,160,67,.2);
    border-radius:10px; padding:12px 16px; margin-bottom:16px;
    font-size:.8rem; color:#8b949e; display:flex; align-items:center; gap:8px;
}
.demo-dot { width:7px; height:7px; background:#3fb950;
    border-radius:50%; flex-shrink:0; animation:bkpulse 1.5s infinite; }
@keyframes bkpulse { 0%,100%{opacity:.4} 50%{opacity:1} }

.doc-meta {
    display:flex; gap:16px; flex-wrap:wrap;
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:10px 14px; margin-bottom:16px;
    font-size:.78rem; color:#8b949e;
}
.doc-meta span strong { color:#c9d1d9; }

.score-badge {
    display:inline-block; background:rgba(46,160,67,.12);
    border:1px solid rgba(46,160,67,.3); border-radius:20px;
    padding:5px 14px; font-size:.8rem; color:#3fb950;
    font-weight:600; margin-bottom:14px;
}

.fc {
    background:#0d1117; border:1.5px solid #30363d; border-radius:10px;
    padding:1.2rem; text-align:center; min-height:110px;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    transition:border-color .2s;
}
.fc.flipped { border-color:#3fb950; background:#070f0a; }
.fc-label { font-size:.65rem; color:#484f57; text-transform:uppercase;
    letter-spacing:.7px; margin-bottom:6px; }
.fc-front { font-size:.9rem; font-weight:600; color:#e6edf3; line-height:1.45; }
.fc-back  { font-size:.8rem; color:#7ee787; line-height:1.6; }

.empty-state { text-align:center; padding:3rem 1rem; color:#484f57; }
.empty-state .icon { font-size:3rem; margin-bottom:1rem; }
.empty-state .msg  { font-size:.95rem; }

.stButton > button {
    background:linear-gradient(135deg,#2ea043,#3fb950) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
    padding:.5rem 1.5rem !important; transition:all .2s !important;
}
.stButton > button:hover    { opacity:.88 !important; transform:translateY(-1px) !important; }
.stButton > button:disabled { opacity:.35 !important; transform:none !important; }
[data-testid="stSidebar"]   { background:#0d1117; border-right:1px solid #21262d; }
.stTextArea textarea, .stTextInput input {
    background:#0d1117 !important; border-color:#30363d !important; color:#e6edf3 !important;
}
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AI ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_client() -> genai.Client:
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        st.error("⚠️  Add `GEMINI_API_KEY` in **Settings → Secrets**.")
        st.stop()
    return genai.Client(api_key=key)


def stream_gemini(prompt: str, system: str = ""):
    """Streaming generator → use with st.write_stream."""
    client = get_client()
    sys = system or "You are Bookiee AI: a sharp, encouraging study assistant. Be precise and educational."
    try:
        response = client.models.generate_content_stream(
            model=MODEL,
            contents=f"{sys}\n\n{prompt}",
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n⚠️  Error: {e}"


def call_gemini(prompt: str, system: str = "") -> str:
    """Blocking call — for JSON outputs."""
    client = get_client()
    sys = system or "Return ONLY raw valid JSON. No markdown, no explanation, no extra text."
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"{sys}\n\n{prompt}",
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return "".join(
            p.text for p in response.candidates[0].content.parts if hasattr(p, "text")
        ).strip()
    except Exception as e:
        return f"Error: {e}"


def call_gemini_json(prompt: str):
    """Returns parsed JSON list or dict, handles both arrays and objects."""
    raw = call_gemini(prompt)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    # Try array first, then object
    for opener, closer in [('[', ']'), ('{', '}')]:
        s = cleaned.find(opener)
        e = cleaned.rfind(closer) + 1
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
#  DOCUMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages[:40])
    return uploaded_file.read().decode("utf-8", errors="ignore")


def doc_metadata(text: str) -> dict:
    words = len(text.split())
    return {
        "words":     words,
        "chars":     len(text),
        "read_mins": max(1, round(words / 238)),
    }


def ctx(text: str) -> str:
    return text[:MAX_CHARS]


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "doc_text":        "",
    "is_demo":         False,
    "analysis":        None,
    "chat_history":    [],
    "quiz_data":       [],
    "quiz_answers":    {},
    "fc_data":         [],
    "fc_flipped":      set(),
    "simplify_text":   "",
    "docs_analyzed":   0,
    "questions_asked": 0,
    "quizzes_taken":   0,
    "session_start":   datetime.now().strftime("%H:%M"),
    "_pending_q":      "",
    "_last_paste":     "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_doc_state():
    for k in ["analysis", "chat_history", "quiz_data", "quiz_answers",
              "fc_data", "fc_flipped", "simplify_text"]:
        st.session_state[k] = DEFAULTS[k]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📚 Bookiee AI")
    st.caption(f"v{APP_VERSION} · Gemini 2.5 Flash")
    st.divider()

    # Demo mode
    st.markdown("**🎯 No document? Try the demo**")
    if st.button("Load Demo Article", use_container_width=True):
        st.session_state.doc_text = DEMO_TEXT
        st.session_state.is_demo  = True
        reset_doc_state()
        st.toast("Demo article loaded!", icon="📖")
        st.rerun()

    st.markdown("**📂 Upload a file**")
    uploaded = st.file_uploader(
        "PDF, TXT, or MD",
        type=["txt", "pdf", "md"],
        label_visibility="collapsed",
    )
    if uploaded:
        with st.spinner("Reading file…"):
            extracted = extract_text(uploaded)
        if extracted.strip():
            st.session_state.doc_text = extracted
            st.session_state.is_demo  = False
            reset_doc_state()
            st.toast(f"✅ {uploaded.name} loaded!", icon="📄")
        else:
            st.error("Couldn't extract text. Try pasting directly.")

    st.markdown("**✏️ Or paste text**")
    pasted = st.text_area(
        "", height=140,
        placeholder="Paste article, chapter, notes…",
        label_visibility="collapsed",
        key="paste_area",
    )
    if pasted and pasted != st.session_state._last_paste:
        st.session_state.doc_text   = pasted
        st.session_state.is_demo    = False
        st.session_state._last_paste = pasted
        reset_doc_state()

    has_doc = len(st.session_state.doc_text.strip()) > 80

    if has_doc:
        meta = doc_metadata(st.session_state.doc_text)
        st.markdown(
            f"<div class='doc-meta'>"
            f"<span>📝 <strong>{meta['words']:,}</strong> words</span>"
            f"<span>⏱ <strong>~{meta['read_mins']} min</strong> read</span>"
            f"<span>💾 <strong>{meta['chars']:,}</strong> chars</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear everything", use_container_width=True):
            st.session_state.doc_text  = ""
            st.session_state.is_demo   = False
            st.session_state._last_paste = ""
            reset_doc_state()
            st.rerun()
    else:
        st.caption("Load demo or add a document to begin.")

    # Session stats
    st.divider()
    st.markdown("**📊 This Session**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Docs",    st.session_state.docs_analyzed)
    c2.metric("Q&As",    st.session_state.questions_asked)
    c3.metric("Quizzes", st.session_state.quizzes_taken)
    st.caption(f"Started {st.session_state.session_start}")

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Upload** PDF / .txt / .md — or paste text
        2. **Analyze** → Summary, key points, concepts
        3. **Chat** → Ask anything about your document
        4. **Study** → Quiz, flashcards, or simplified explanation

        Free tier · Powered by **Gemini 2.5 Flash**
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bk-header">
  <h1>📚 Bookiee AI</h1>
  <p>Upload any document → Understand it deeply, study it fast.</p>
</div>
""", unsafe_allow_html=True)

document_ctx = ctx(st.session_state.doc_text)

tab1, tab2, tab3 = st.tabs([
    "📄  Analyze Document",
    "💬  Ask Questions",
    "🧠  Study Mode",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not has_doc:
        st.markdown("""<div class='empty-state'>
            <div class='icon'>📂</div>
            <div class='msg'>Load the demo article or upload your own to get started.</div>
        </div>""", unsafe_allow_html=True)
    else:
        if st.session_state.is_demo:
            st.markdown("""<div class='demo-banner'>
                <div class='demo-dot'></div>
                <span>Demo mode — sample article on <strong>Deep Work</strong>.
                Upload your own document from the sidebar anytime.</span>
            </div>""", unsafe_allow_html=True)

        already_analyzed = st.session_state.analysis is not None

        if not already_analyzed:
            if st.button("✨ Analyze Document", use_container_width=True):
                prog = st.progress(0, "Starting…")

                # ── Summary (streamed live) ────────────────────────────────
                prog.progress(10, "Generating summary…")
                holder = st.empty()
                full_summary = ""
                for chunk in stream_gemini(
                    f"Write a clear, insightful summary in 3-4 paragraphs:\n\n{document_ctx}"
                ):
                    full_summary += chunk
                    holder.markdown(
                        f'<div class="bk-card"><div class="bk-card-title">📝 Summary</div>'
                        f'<div class="bk-card-body">{full_summary}▌</div></div>',
                        unsafe_allow_html=True,
                    )
                holder.empty()

                # ── Key points + concepts (parallel JSON calls) ───────────
                prog.progress(55, "Extracting key points & concepts…")
                key_points = call_gemini_json(
                    f"Extract exactly 7 key ideas. "
                    f"Return a JSON array of strings, each ≤ 15 words:\n\n{document_ctx}"
                )
                concepts = call_gemini_json(
                    f"Identify 5 important concepts. Return a JSON array of objects "
                    f"with 'term' and 'definition' (one sentence) fields:\n\n{document_ctx}"
                )

                prog.progress(100, "Done!")
                time.sleep(0.25)
                prog.empty()

                st.session_state.analysis = {
                    "summary":    full_summary,
                    "key_points": key_points or [],
                    "concepts":   concepts   or [],
                }
                st.session_state.docs_analyzed += 1
                st.toast("Analysis complete!", icon="✅")
                st.rerun()

        if st.session_state.analysis:
            a = st.session_state.analysis

            # Action row
            col_rerun, col_export = st.columns(2)
            with col_rerun:
                if st.button("🔄 Re-analyze", use_container_width=True):
                    st.session_state.analysis = None
                    st.rerun()
            with col_export:
                md = (
                    f"# Bookiee AI — Document Analysis\n\n"
                    f"## Summary\n{a['summary']}\n\n"
                    f"## Key Points\n" + "\n".join(f"- {p}" for p in a["key_points"]) +
                    f"\n\n## Key Concepts\n" +
                    "".join(f"**{c.get('term','')}** — {c.get('definition','')}\n\n"
                            for c in a["concepts"])
                )
                st.download_button(
                    "⬇️ Export as Markdown", data=md,
                    file_name="bookiee_analysis.md", mime="text/markdown",
                    use_container_width=True,
                )

            st.divider()

            # Summary
            st.markdown(
                f'<div class="bk-card"><div class="bk-card-title">📝 Summary</div>'
                f'<div class="bk-card-body">{a["summary"]}</div></div>',
                unsafe_allow_html=True,
            )

            # Key Points
            pills = " ".join(
                f'<span class="bk-pill">• {p}</span>'
                for p in a["key_points"]
            )
            st.markdown(
                f'<div class="bk-card"><div class="bk-card-title">🔑 Key Points</div>'
                f'{pills}</div>',
                unsafe_allow_html=True,
            )

            # Concepts + deep-dive
            st.markdown(
                '<div class="bk-card"><div class="bk-card-title">💡 Key Concepts</div>',
                unsafe_allow_html=True,
            )
            for c in a["concepts"]:
                term = c.get("term", "")
                defi = c.get("definition", "")
                st.markdown(
                    f'<div class="bk-concept">'
                    f'<div class="bk-concept-term">{term}</div>'
                    f'<div class="bk-concept-def">{defi}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"🔍 Deep dive into: {term}"):
                    st.write_stream(stream_gemini(
                        f"Give a richer 2-3 paragraph explanation of '{term}' in the context "
                        f"of the document below. Include a real-world example or analogy:\n\n{document_ctx}"
                    ))
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not has_doc:
        st.markdown("""<div class='empty-state'>
            <div class='icon'>💬</div>
            <div class='msg'>Add a document first, then ask anything about it.</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Quick starters
        if not st.session_state.chat_history:
            st.markdown("**Quick questions:**")
            q1, q2, q3 = st.columns(3)
            starters = [
                "What's the main argument?",
                "Explain the key concept simply",
                "Give me 3 actionable takeaways",
            ]
            for col, q in zip([q1, q2, q3], starters):
                if col.button(q, use_container_width=True, key=f"qs_{q[:6]}"):
                    st.session_state._pending_q = q
                    st.rerun()

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask anything about your document…")
        if st.session_state._pending_q:
            user_input = st.session_state._pending_q
            st.session_state._pending_q = ""

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.questions_asked += 1

            with st.chat_message("assistant", avatar="📚"):
                answer = st.write_stream(stream_gemini(
                    f"Answer clearly based only on the context below. "
                    f"If the answer isn't there, say so honestly.\n\n"
                    f"Context:\n{document_ctx}\n\nQuestion: {user_input}"
                ))

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Follow-up suggestions
            suggestions = call_gemini_json(
                f"Suggest 3 short follow-up questions based on this Q&A. "
                f"Return a JSON array of strings.\n\nQ: {user_input}\nA: {str(answer)[:300]}"
            )
            if suggestions and isinstance(suggestions, list):
                st.markdown("**Follow-up:**")
                s_cols = st.columns(len(suggestions[:3]))
                for i, (col, sug) in enumerate(zip(s_cols, suggestions[:3])):
                    if col.button(sug, key=f"sug_{i}_{len(st.session_state.chat_history)}",
                                  use_container_width=True):
                        st.session_state._pending_q = sug
                        st.rerun()

            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STUDY MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not has_doc:
        st.markdown("""<div class='empty-state'>
            <div class='icon'>🧠</div>
            <div class='msg'>Add a document to generate quizzes, flashcards, and explanations.</div>
        </div>""", unsafe_allow_html=True)
    else:
        mode = st.radio(
            "",
            ["📝  Quiz", "🃏  Flashcards", "✏️  Simplify"],
            horizontal=True, label_visibility="collapsed",
        )

        # ── QUIZ ──────────────────────────────────────────────────────────
        if mode == "📝  Quiz":
            diff = st.select_slider(
                "Difficulty", options=["Easy", "Medium", "Hard"], value="Medium"
            )
            if st.button("🚀 Generate Quiz", use_container_width=True):
                with st.spinner("Building your quiz…"):
                    hint = {
                        "Easy":   "straightforward recall",
                        "Medium": "comprehension and application",
                        "Hard":   "analysis and critical thinking",
                    }[diff]
                    data = call_gemini_json(
                        f"Create 5 {hint} multiple-choice questions from this text. "
                        f"Return a JSON array of objects with: "
                        f"'question' (string), 'options' (array of 4 strings starting A. B. C. D.), "
                        f"'answer' (single letter A B C or D), 'explanation' (≤ 25 words).\n\n{document_ctx}"
                    )
                    if data and isinstance(data, list):
                        st.session_state.quiz_data    = data
                        st.session_state.quiz_answers = {}
                        st.session_state.quizzes_taken += 1
                        st.toast("Quiz ready!", icon="📝")
                    else:
                        st.error("Quiz generation failed. Try again or use a longer document.")

            if st.session_state.quiz_data:
                total   = len(st.session_state.quiz_data)
                answered = len(st.session_state.quiz_answers)
                correct  = sum(
                    1 for i, q in enumerate(st.session_state.quiz_data)
                    if st.session_state.quiz_answers.get(i) == q.get("answer", " ").strip()[0]
                )

                st.markdown(
                    f'<div class="score-badge">📊 {correct}/{total} correct · {answered} answered</div>',
                    unsafe_allow_html=True,
                )

                if answered == total and total > 0:
                    result_md = f"# Quiz Results — Bookiee AI\nScore: {correct}/{total}\n\n"
                    for i, q in enumerate(st.session_state.quiz_data):
                        given = st.session_state.quiz_answers.get(i, "?")
                        result_md += (
                            f"**Q{i+1}.** {q.get('question','')}\n"
                            f"Your answer: {given} | Correct: {q.get('answer','')}\n"
                            f"Explanation: {q.get('explanation','')}\n\n"
                        )
                    st.download_button(
                        "⬇️ Export Results", data=result_md,
                        file_name="bookiee_quiz.md", mime="text/markdown",
                    )

                for i, q in enumerate(st.session_state.quiz_data):
                    done = i in st.session_state.quiz_answers
                    given_ans = st.session_state.quiz_answers.get(i, "")
                    correct_ans = q.get("answer", " ").strip()[0]
                    icon = "✅" if done and given_ans == correct_ans else ("❌" if done else "○")
                    with st.expander(f"{icon}  Q{i+1}. {q.get('question','')}", expanded=not done):
                        chosen = st.radio(
                            "Select answer:",
                            q.get("options", []),
                            key=f"qr_{i}",
                            label_visibility="collapsed",
                        )
                        if not done:
                            if st.button("Submit", key=f"qsub_{i}"):
                                st.session_state.quiz_answers[i] = (chosen or " ")[0]
                                st.rerun()
                        else:
                            if given_ans == correct_ans:
                                st.success(f"✅ Correct! Answer is **{correct_ans}**")
                            else:
                                st.error(f"❌ You chose **{given_ans}** · Correct: **{correct_ans}**")
                            st.caption(f"💡 {q.get('explanation','')}")

        # ── FLASHCARDS ─────────────────────────────────────────────────────
        elif mode == "🃏  Flashcards":
            n = st.slider("Number of cards", 4, 16, 8, step=2)
            if st.button(" Generate Flashcards", use_container_width=True):
                with st.spinner(f"Creating {n} flashcards…"):
                    data = call_gemini_json(
                        f"Create exactly {n} flashcards from this text. "
                        f"Return a JSON array of objects with 'front' (term/question) "
                        f"and 'back' (definition/answer) fields.\n\n{document_ctx}"
                    )
                    if data and isinstance(data, list):
                        st.session_state.fc_data    = data
                        st.session_state.fc_flipped = set()
                        st.toast(f"{len(data)} flashcards ready!", icon="🃏")
                    else:
                        st.error("Flashcard generation failed. Try again.")

            if st.session_state.fc_data:
                total_fc = len(st.session_state.fc_data)
                revealed = len(st.session_state.fc_flipped)
                st.markdown(
                    f'<div class="score-badge">🃏 {revealed}/{total_fc} revealed</div>',
                    unsafe_allow_html=True,
                )
                fa, fb = st.columns(2)
                if fa.button("Flip all",  use_container_width=True):
                    st.session_state.fc_flipped = set(range(total_fc)); st.rerun()
                if fb.button("Reset all", use_container_width=True):
                    st.session_state.fc_flipped = set(); st.rerun()

                cols = st.columns(2)
                for i, card in enumerate(st.session_state.fc_data):
                    flipped = i in st.session_state.fc_flipped
                    content = card.get("back" if flipped else "front", "")
                    css_cls = "fc flipped" if flipped else "fc"
                    txt_cls = "fc-back"     if flipped else "fc-front"
                    label   = "BACK ↩"      if flipped else "FRONT"
                    with cols[i % 2]:
                        st.markdown(
                            f'<div class="{css_cls}">'
                            f'<div class="fc-label">{label}</div>'
                            f'<div class="{txt_cls}">{content}</div></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("Flip 🔄", key=f"fc_{i}", use_container_width=True):
                            if flipped: st.session_state.fc_flipped.discard(i)
                            else:       st.session_state.fc_flipped.add(i)
                            st.rerun()

        # ── SIMPLIFY ───────────────────────────────────────────────────────
        elif mode == "✏️  Simplify":
            audience = st.radio(
                "Explain to me as a:",
                ["Complete beginner", "Curious teenager", "Busy professional"],
                horizontal=True,
            )
            if st.button("🚀 Simplify", use_container_width=True):
                persona = {
                    "Complete beginner":   "complete beginner — use simple words, analogies, and relatable examples",
                    "Curious teenager":    "curious 16-year-old — be engaging, use pop culture references, keep it lively",
                    "Busy professional":   "busy professional who needs the core insight in 3 minutes — be dense, direct, actionable",
                }[audience]
                holder     = st.empty()
                full_text  = ""
                for chunk in stream_gemini(
                    f"Explain this text to a {persona}:\n\n{document_ctx}"
                ):
                    full_text += chunk
                    holder.markdown(
                        f'<div class="bk-card">'
                        f'<div class="bk-card-title">✏️ For: {audience}</div>'
                        f'<div class="bk-card-body">{full_text}▌</div></div>',
                        unsafe_allow_html=True,
                    )
                holder.empty()
                st.session_state.simplify_text = full_text

            if st.session_state.simplify_text:
                st.markdown(
                    f'<div class="bk-card">'
                    f'<div class="bk-card-title">✏️ Simplified Explanation</div>'
                    f'<div class="bk-card-body">{st.session_state.simplify_text}</div></div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇️ Export Explanation",
                    data=st.session_state.simplify_text,
                    file_name="bookiee_simplified.txt",
                    mime="text/plain",
                )


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:.75rem;color:#484f57'>"
    "Bookiee AI · Streamlit + Gemini 2.5 Flash · "
    "<a href='https://github.com' style='color:#3fb950;text-decoration:none'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True,
)
