"""
app.py - Bookiee AI orchestrator (v1.2)

This file contains ONLY layout and interaction logic.
All business logic lives in: state.py, ai_engine.py, chunker_utils.py, prompts.py
"""

import streamlit as st
import pdfplumber
import hashlib
import io
import time

import state
import ai_engine as ai
import chunker_utils as chu
import prompts

# ── Init session state FIRST, before any st calls that reference it ────────────
state.init()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bookiee AI",
    page_icon="\U0001f4da",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "1.2.0"

DEMO_TEXT = (
    "The Science of Deep Work: How Focus Became the New Superpower\n\n"
    "In an age of infinite distraction, the ability to concentrate deeply on demanding "
    "tasks has become one of the rarest and most valuable skills in the modern economy. "
    "Cal Newport coined the term deep work to describe professional activities performed "
    "in a state of distraction-free concentration that push your cognitive capabilities "
    "to their limit.\n\n"
    "The Neurological Case for Focus\n"
    "When you spend time in a state of deep work, two things happen. First, myelin, a "
    "white tissue around neurons, grows thicker with use, making neural pathways fire "
    "faster. Second, the default mode network quiets down, allowing the prefrontal cortex "
    "to operate at full capacity. The result is structurally better thinking.\n\n"
    "Shallow Work: The Silent Productivity Killer\n"
    "Contrast deep work with shallow work - non-cognitively demanding tasks performed "
    "while distracted. Emails, status meetings, and social scrolling all qualify. The "
    "problem is that shallow work creates the illusion of productivity. Studies from UC "
    "Irvine found it takes 23 minutes to regain concentration after an interruption.\n\n"
    "The Four Philosophies\n"
    "Newport identifies four approaches: The Monastic (eliminate all shallow obligations), "
    "The Bimodal (alternate deep and shallow seasons), The Rhythmic (daily deep work "
    "ritual), and The Journalistic (fit deep work wherever you can).\n\n"
    "Training the Attention Muscle\n"
    "Deep work is a skill that strengthens with use. Your capacity for concentration is "
    "like a muscle: it atrophies without training. Start with 1-hour blocks and build "
    "toward 4-hour sessions. Boredom is the gym.\n\n"
    "The Economic Argument\n"
    "In the new economy, two groups thrive: those who master hard things quickly, and "
    "those who produce at an elite level. Both depend on deep work. In a world where "
    "attention is for sale, those who own theirs have a decisive advantage."
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.bk-header{text-align:center;padding:1.5rem 0 .75rem}
.bk-header h1{font-family:'Playfair Display',serif;font-size:2.5rem;
  background:linear-gradient(135deg,#2ea043,#3fb950);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem}
.bk-header p{color:#8b949e;font-size:.9rem;margin:0}
.bk-card{background:#161b22;border:1px solid #30363d;border-radius:12px;
  padding:1.25rem 1.5rem;margin-bottom:1rem}
.bk-card-title{font-size:.7rem;font-weight:700;color:#3fb950;text-transform:uppercase;
  letter-spacing:.9px;margin-bottom:.75rem}
.bk-card-body{font-size:.875rem;color:#c9d1d9;line-height:1.75}
.bk-pill{display:inline-block;background:#21262d;border:1px solid #30363d;
  border-radius:20px;padding:5px 13px;font-size:.78rem;color:#c9d1d9;margin:3px}
.bk-concept{border-left:2.5px solid #3fb950;padding:10px 14px;margin-bottom:10px;
  background:#0d1117;border-radius:0 8px 8px 0}
.bk-concept-term{color:#e6edf3;font-size:.875rem;font-weight:600;margin-bottom:3px}
.bk-concept-def{color:#8b949e;font-size:.8rem;line-height:1.55}
.demo-banner{background:rgba(46,160,67,.08);border:1px solid rgba(46,160,67,.2);
  border-radius:10px;padding:12px 16px;margin-bottom:16px;
  font-size:.8rem;color:#8b949e;display:flex;align-items:center;gap:8px}
.demo-dot{width:7px;height:7px;background:#3fb950;border-radius:50%;
  flex-shrink:0;animation:bkp 1.5s infinite}
@keyframes bkp{0%,100%{opacity:.4}50%{opacity:1}}
.doc-meta{display:flex;gap:16px;flex-wrap:wrap;background:#161b22;
  border:1px solid #30363d;border-radius:8px;padding:10px 14px;
  margin-bottom:16px;font-size:.78rem;color:#8b949e}
.doc-meta span strong{color:#c9d1d9}
.chunk-badge{background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.25);
  border-radius:20px;padding:4px 12px;font-size:.72rem;color:#d97706;
  margin-bottom:12px;display:inline-block}
.score-badge{display:inline-block;background:rgba(46,160,67,.12);
  border:1px solid rgba(46,160,67,.3);border-radius:20px;padding:5px 14px;
  font-size:.8rem;color:#3fb950;font-weight:600;margin-bottom:14px}
.fc{background:#0d1117;border:1.5px solid #30363d;border-radius:10px;
  padding:1.2rem;text-align:center;min-height:110px;display:flex;
  flex-direction:column;align-items:center;justify-content:center;transition:border-color .2s}
.fc.flipped{border-color:#3fb950;background:#070f0a}
.fc-label{font-size:.65rem;color:#484f57;text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px}
.fc-front{font-size:.9rem;font-weight:600;color:#e6edf3;line-height:1.45}
.fc-back{font-size:.8rem;color:#7ee787;line-height:1.6}
.empty-state{text-align:center;padding:3rem 1rem;color:#484f57}
.empty-icon{font-size:3rem;margin-bottom:1rem}
.stButton>button{background:linear-gradient(135deg,#2ea043,#3fb950)!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  font-weight:600!important;padding:.5rem 1.5rem!important;transition:opacity .15s!important}
.stButton>button:hover{opacity:.85!important}
.stButton>button:disabled{opacity:.35!important}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d}
.stTextArea textarea{background:#0d1117!important;border-color:#30363d!important;color:#e6edf3!important}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _extract(f) -> str:
    if f.name.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(f.read())) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages[:50])
    return f.read().decode("utf-8", errors="ignore")


def _empty(icon: str, msg: str):
    st.markdown(
        f"<div class='empty-state'><div class='empty-icon'>{icon}</div>"
        f"<div>{msg}</div></div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### \U0001f4da Bookiee AI")
    st.caption(f"v{APP_VERSION}  \u00b7  Gemini 2.5 Flash")
    st.divider()

    # -- Demo
    st.markdown("**\U0001f3af Try a demo**")
    if st.button("Load Demo Article", use_container_width=True):
        if state.set_document(DEMO_TEXT, source_id="__demo__",
                              filename="demo.txt", is_demo=True):
            st.toast("Demo loaded!", icon="\U0001f4d6")
        st.rerun()

    # -- File upload
    st.markdown("**\U0001f4c2 Upload a file**")
    uploaded = st.file_uploader("PDF / TXT / MD", type=["txt","pdf","md"],
                                label_visibility="collapsed")
    if uploaded:
        # KEY FIX: file_id is stable across reruns for the same file.
        # set_document() is a no-op when source_id matches -> no state reset.
        file_id = f"file::{uploaded.name}::{uploaded.size}"
        if file_id != st.session_state.get(state.DOC_SRC_ID, ""):
            with st.spinner("Extracting..."):
                text = _extract(uploaded)
            if text.strip():
                state.set_document(text, source_id=file_id,
                                   filename=uploaded.name)
                st.toast(f"{uploaded.name} loaded!", icon="\U0001f4c4")
                st.rerun()
            else:
                st.error("Could not extract text. Try pasting below.")

    # -- Paste
    st.markdown("**\u270f\ufe0f Or paste text**")
    pasted = st.text_area("", height=130,
                          placeholder="Paste article, chapter, notes...",
                          label_visibility="collapsed",
                          key="bk_paste_area")
    if pasted:
        paste_id = f"paste::{hashlib.md5(pasted.encode()).hexdigest()[:10]}"
        if paste_id != st.session_state.get(state.DOC_SRC_ID, ""):
            state.set_document(pasted, source_id=paste_id)

    has_doc  = state.has_doc()
    doc_text = st.session_state.get(state.DOC_TEXT, "")
    d_hash   = chu.make_hash(doc_text) if has_doc else ""
    long_doc = chu.is_long(doc_text) if has_doc else False

    if has_doc:
        meta = chu.doc_meta(doc_text)
        st.markdown(
            f"<div class='doc-meta'>"
            f"<span>\U0001f4dd <strong>{meta['words']:,}</strong> words</span>"
            f"<span>\u23f1 ~<strong>{meta['read_mins']} min</strong></span>"
            f"<span>\U0001f4be <strong>{meta['chars']:,}</strong> chars</span>"
            f"</div>", unsafe_allow_html=True)
        if long_doc:
            nc = len(chu.chunk_doc(doc_text))
            st.markdown(
                f'<div class="chunk-badge">'
                f'\U0001f4e6 Long doc \u00b7 {nc} sections \u00b7 smart chunking on'
                f'</div>', unsafe_allow_html=True)
        if st.button("Clear everything", use_container_width=True):
            st.session_state[state.DOC_TEXT]    = ""
            st.session_state[state.DOC_SRC_ID]  = ""
            st.session_state[state.IS_DEMO]     = False
            state._reset_doc_derived()
            st.rerun()
    else:
        st.caption("Load demo or add a document to begin.")

    # Session stats
    st.divider()
    st.markdown("**\U0001f4ca Session**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Docs",    st.session_state.get(state.DOCS_ANALYZED, 0))
    c2.metric("Q&As",    st.session_state.get(state.QS_ASKED, 0))
    c3.metric("Quizzes", st.session_state.get(state.QUIZZES_TAKEN, 0))
    st.caption(f"Started {st.session_state.get(state.SESSION_START, '')}")

    st.divider()
    with st.expander("How it works"):
        st.markdown("""
1. **Upload** PDF / .txt / .md or paste text
2. **Analyze** \u2192 summary, key points, concepts  
   *(1 API call \u2014 was 3)*
3. **Chat** \u2192 ask anything; long docs use smart section matching
4. **Study** \u2192 quiz, flashcards, simplified explanation

All results are **cached** for 2 hours \u2014 zero redundant calls.
Long docs are **chunked** and synthesized automatically.
        """)
    with st.expander("AI diagnostics"):
        ai.render_diagnostics()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bk-header">
  <h1>\U0001f4da Bookiee AI</h1>
  <p>Upload any document \u2192 understand it deeply \u00b7 study it fast.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "\U0001f4c4  Analyze",
    "\U0001f4ac  Ask Questions",
    "\U0001f9e0  Study Mode",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 \u2014 ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not has_doc:
        _empty("\U0001f4c2", "Load the demo or upload a document to begin.")
    else:
        if st.session_state.get(state.IS_DEMO):
            st.markdown(
                "<div class='demo-banner'><div class='demo-dot'></div>"
                "<span>Demo mode \u2014 <strong>Deep Work</strong> by Cal Newport. "
                "Upload your own from the sidebar.</span></div>",
                unsafe_allow_html=True)

        analysis = st.session_state.get(state.ANALYSIS)

        # -- Generate
        if analysis is None:
            if st.button("Analyze Document", use_container_width=True):
                prog   = st.progress(0, "Starting...")
                sum_ph = st.empty()

                if long_doc:
                    prog.progress(5, "Chunking document...")
                    sections = ai.get_section_summaries(d_hash, doc_text)

                    prog.progress(60, "Synthesizing summary...")
                    full_summary = ""
                    for chunk in ai.stream_call(
                        prompts.synthesize_summary(sections), feature="summary"
                    ):
                        full_summary += chunk
                        sum_ph.markdown(
                            f'<div class="bk-card"><div class="bk-card-title">Summary</div>'
                            f'<div class="bk-card-body">{full_summary}</div></div>',
                            unsafe_allow_html=True)
                    sum_ph.empty()

                    prog.progress(80, "Extracting points and concepts...")
                    pc = ai.get_points_and_concepts(
                        d_hash, "\n\n".join(sections)[:chu.MAX_CHARS])

                else:
                    ctx = doc_text[:chu.MAX_CHARS]

                    prog.progress(10, "Generating summary...")
                    full_summary = ""
                    for chunk in ai.stream_call(
                        prompts.summary(ctx), feature="summary"
                    ):
                        full_summary += chunk
                        sum_ph.markdown(
                            f'<div class="bk-card"><div class="bk-card-title">Summary</div>'
                            f'<div class="bk-card-body">{full_summary}</div></div>',
                            unsafe_allow_html=True)
                    sum_ph.empty()

                    prog.progress(65, "Extracting points and concepts...")
                    pc = ai.get_points_and_concepts(d_hash, ctx)

                prog.progress(100, "Done!")
                time.sleep(0.2)
                prog.empty()

                st.session_state[state.ANALYSIS] = {
                    "summary":    full_summary,
                    "key_points": pc.get("key_points", []),
                    "concepts":   pc.get("concepts", []),
                }
                st.session_state[state.DOCS_ANALYZED] = (
                    st.session_state.get(state.DOCS_ANALYZED, 0) + 1
                )
                st.toast("Analysis complete!", icon="\u2705")
                st.rerun()

        # -- Render
        if st.session_state.get(state.ANALYSIS):
            a = st.session_state[state.ANALYSIS]

            col_r, col_e = st.columns(2)
            with col_r:
                if st.button("Re-analyze", use_container_width=True):
                    st.session_state[state.ANALYSIS] = None
                    st.rerun()
            with col_e:
                md = ("# Bookiee AI \u2014 Analysis\n\n## Summary\n" +
                      a["summary"] + "\n\n## Key Points\n" +
                      "\n".join(f"- {p}" for p in a["key_points"]) +
                      "\n\n## Key Concepts\n" +
                      "".join(f"**{c.get('term','')}** \u2014 {c.get('definition','')}\n\n"
                              for c in a["concepts"]))
                st.download_button("Export Markdown", data=md,
                                   file_name="bookiee_analysis.md",
                                   mime="text/markdown", use_container_width=True)

            if long_doc:
                nc = len(chu.chunk_doc(doc_text))
                st.markdown(
                    f'<div class="chunk-badge">'
                    f'Full-document analysis across {nc} sections</div>',
                    unsafe_allow_html=True)

            st.divider()

            # Summary card
            st.markdown(
                f'<div class="bk-card"><div class="bk-card-title">Summary</div>'
                f'<div class="bk-card-body">{a["summary"]}</div></div>',
                unsafe_allow_html=True)

            # Key points
            pills = " ".join(
                f'<span class="bk-pill">\u2022 {p}</span>'
                for p in a["key_points"])
            st.markdown(
                f'<div class="bk-card"><div class="bk-card-title">Key Points</div>'
                f'{pills}</div>', unsafe_allow_html=True)

            # Concepts + lazy deep dives
            st.markdown(
                '<div class="bk-card"><div class="bk-card-title">Key Concepts</div>',
                unsafe_allow_html=True)
            deep_dives = st.session_state.get(state.DEEP_DIVES, {})

            for c in a["concepts"]:
                term = c.get("term", "")
                defi = c.get("definition", "")
                st.markdown(
                    f'<div class="bk-concept">'
                    f'<div class="bk-concept-term">{term}</div>'
                    f'<div class="bk-concept-def">{defi}</div></div>',
                    unsafe_allow_html=True)

                with st.expander(f"Deep dive: {term}"):
                    if term in deep_dives:
                        # Cached result \u2014 render instantly, zero API cost
                        st.markdown(deep_dives[term])
                    else:
                        # Lazy: never executes automatically on expander open
                        if st.button(f"Generate deep dive",
                                     key=f"dd_{term}_{d_hash}"):
                            with st.spinner(f"Researching {term}..."):
                                dive_ctx = (
                                    chu.smart_chat_ctx(doc_text, term)
                                    if long_doc else doc_text[:chu.MAX_CHARS]
                                )
                                result = ai.get_deep_dive(d_hash, term, dive_ctx)
                            if result:
                                deep_dives[term] = result
                                st.session_state[state.DEEP_DIVES] = deep_dives
                                st.markdown(result)

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 \u2014 CHAT
#  Rerun minimization:
#    - No explicit st.rerun() after chat answers (widget interaction reruns naturally)
#    - Follow-up buttons set PENDING_Q; the next rerun processes it at the top
#      of this block, BEFORE st.chat_input(), so no extra rerun cycle is needed.
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not has_doc:
        _empty("\U0001f4ac", "Add a document first, then ask anything about it.")
    else:
        if long_doc:
            st.markdown(
                '<div class="chunk-badge">'
                'Long doc \u2014 relevant sections retrieved per question</div>',
                unsafe_allow_html=True)

        chat_history  = st.session_state.get(state.CHAT_HISTORY, [])
        last_followups = st.session_state.get(state.LAST_FOLLOWUPS, [])

        # Quick starters (only when empty)
        if not chat_history:
            st.markdown("**Quick questions:**")
            qa, qb, qc = st.columns(3)
            for col, q in zip([qa, qb, qc], [
                "What is the main argument?",
                "Explain the key concept simply",
                "Give me 3 actionable takeaways",
            ]):
                if col.button(q, use_container_width=True, key=f"qs_{q[:8]}"):
                    st.session_state[state.PENDING_Q] = q
                    # No st.rerun() needed \u2014 the button click triggers one automatically

        # Render full history
        for i, msg in enumerate(chat_history):
            with st.chat_message(msg["role"],
                                 avatar="\U0001f9d1" if msg["role"] == "user" else "\U0001f4da"):
                st.markdown(msg["content"])
            # Attach follow-up buttons to the last assistant message
            if (msg["role"] == "assistant"
                    and i == len(chat_history) - 1
                    and last_followups):
                st.markdown("**You might also ask:**")
                fu_cols = st.columns(min(3, len(last_followups)))
                for j, (col, sug) in enumerate(zip(fu_cols, last_followups[:3])):
                    if col.button(sug, key=f"fu_{j}_{len(chat_history)}",
                                  use_container_width=True):
                        st.session_state[state.PENDING_Q] = sug
                        # Button click triggers rerun; PENDING_Q consumed below

        # Consume PENDING_Q first, then st.chat_input
        # This eliminates the extra st.rerun() call that was previously needed
        user_input = st.chat_input("Ask anything about your document...")
        if not user_input:
            pending = st.session_state.get(state.PENDING_Q, "")
            if pending:
                user_input = pending
                st.session_state[state.PENDING_Q] = ""

        if user_input:
            chat_history.append({"role": "user", "content": user_input})
            st.session_state[state.CHAT_HISTORY]   = chat_history
            st.session_state[state.QS_ASKED]       = (
                st.session_state.get(state.QS_ASKED, 0) + 1
            )

            ctx = chu.smart_chat_ctx(doc_text, user_input)
            with st.chat_message("assistant", avatar="\U0001f4da"):
                answer = st.write_stream(ai.answer_question(ctx, user_input))

            chat_history.append({"role": "assistant", "content": answer})
            st.session_state[state.CHAT_HISTORY] = chat_history

            # Follow-up suggestions (cached by Q+A hash)
            qa_hash = hashlib.md5(
                (user_input + str(answer)[:200]).encode()
            ).hexdigest()[:8]
            suggestions = ai.get_follow_ups(
                qa_hash, user_input, str(answer)[:400]
            )
            st.session_state[state.LAST_FOLLOWUPS] = suggestions
            # No st.rerun() \u2014 Streamlit will rerender on the next natural interaction.
            # The follow-up buttons are attached to the history render above
            # and will appear on the next rerun.

        if chat_history:
            if st.button("Clear chat"):
                st.session_state[state.CHAT_HISTORY]   = []
                st.session_state[state.LAST_FOLLOWUPS] = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 \u2014 STUDY MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not has_doc:
        _empty("\U0001f9e0", "Add a document to generate study materials.")
    else:
        mode = st.radio("", ["\U0001f4dd  Quiz", "\U0001f0cf  Flashcards", "\u270f\ufe0f  Simplify"],
                        horizontal=True, label_visibility="collapsed")

        # ── QUIZ ──────────────────────────────────────────────────────────────
        if mode == "\U0001f4dd  Quiz":
            diff = st.select_slider("Difficulty",
                                    options=["Easy", "Medium", "Hard"],
                                    value="Medium")

            if st.button("Generate Quiz", use_container_width=True):
                with st.spinner("Building quiz..."):
                    if long_doc:
                        sections = ai.get_section_summaries(d_hash, doc_text)
                        ctx = "\n\n".join(sections)[:chu.MAX_CHARS]
                    else:
                        ctx = doc_text[:chu.MAX_CHARS]
                    questions = ai.get_quiz(d_hash, ctx, diff)
                    if questions:
                        st.session_state[state.QUIZ_DATA]    = questions
                        st.session_state[state.QUIZ_ANSWERS] = {}
                        st.session_state[state.QUIZZES_TAKEN] = (
                            st.session_state.get(state.QUIZZES_TAKEN, 0) + 1
                        )
                        st.toast("Quiz ready!", icon="\U0001f4dd")
                    else:
                        st.error("Quiz generation failed. Try again.")
                # No st.rerun() needed - questions render from session state below

            quiz_data    = st.session_state.get(state.QUIZ_DATA, [])
            quiz_answers = st.session_state.get(state.QUIZ_ANSWERS, {})

            if quiz_data:
                total    = len(quiz_data)
                answered = len(quiz_answers)
                correct  = sum(1 for i, q in enumerate(quiz_data)
                               if quiz_answers.get(i) == q.get("answer", ""))

                st.markdown(
                    f'<div class="score-badge">'
                    f'{correct}/{total} correct \u00b7 {answered} answered</div>',
                    unsafe_allow_html=True)

                if answered == total > 0:
                    rmd = f"# Quiz Results\nScore: {correct}/{total}\n\n"
                    for i, q in enumerate(quiz_data):
                        g = quiz_answers.get(i, "?")
                        rmd += (f"**Q{i+1}.** {q.get('question','')}\n"
                                f"Yours: {g} | Correct: {q.get('answer','')}\n"
                                f"Explanation: {q.get('explanation','')}\n\n")
                    st.download_button("Export Results", data=rmd,
                                       file_name="bookiee_quiz.md",
                                       mime="text/markdown")

                for i, q in enumerate(quiz_data):
                    done  = i in quiz_answers
                    given = quiz_answers.get(i, "")
                    ans   = q.get("answer", "")
                    icon  = "\u2705" if done and given == ans else ("\u274c" if done else "\u25cb")

                    with st.expander(f"{icon}  Q{i+1}. {q.get('question','')[:75]}",
                                     expanded=not done):
                        chosen = st.radio("", q.get("options", []),
                                          key=f"qr_{i}_{d_hash}",
                                          label_visibility="collapsed")
                        if not done:
                            if st.button("Submit", key=f"qs_{i}_{d_hash}"):
                                st.session_state[state.QUIZ_ANSWERS][i] = (
                                    (chosen or " ")[0]
                                )
                                # st.rerun() needed here: the `done` flag was evaluated
                                # at the top of this loop before the button click.
                                # Without rerun, the result won't show until next interaction.
                                st.rerun()
                        else:
                            if given == ans:
                                st.success(f"Correct! Answer: **{ans}**")
                            else:
                                st.error(f"You chose **{given}** \u00b7 Correct: **{ans}**")
                            st.caption(q.get("explanation", ""))

        # ── FLASHCARDS ────────────────────────────────────────────────────────
        elif mode == "\U0001f0cf  Flashcards":
            n = st.slider("Number of cards", 4, 16, 8, step=2)

            if st.button("Generate Flashcards", use_container_width=True):
                with st.spinner(f"Creating {n} flashcards..."):
                    if long_doc:
                        sections = ai.get_section_summaries(d_hash, doc_text)
                        ctx = "\n\n".join(sections)[:chu.MAX_CHARS]
                    else:
                        ctx = doc_text[:chu.MAX_CHARS]
                    cards = ai.get_flashcards(d_hash, ctx, n)
                    if cards:
                        st.session_state[state.FC_DATA]    = cards
                        st.session_state[state.FC_FLIPPED] = set()
                        st.toast(f"{len(cards)} flashcards ready!", icon="\U0001f0cf")
                    else:
                        st.error("Flashcard generation failed. Try again.")

            fc_data    = st.session_state.get(state.FC_DATA, [])
            fc_flipped = st.session_state.get(state.FC_FLIPPED, set())

            if fc_data:
                total_fc = len(fc_data)
                revealed = len(fc_flipped)
                st.markdown(
                    f'<div class="score-badge">{revealed}/{total_fc} revealed</div>',
                    unsafe_allow_html=True)

                fa, fb = st.columns(2)
                if fa.button("Flip all", use_container_width=True):
                    st.session_state[state.FC_FLIPPED] = set(range(total_fc))
                    st.rerun()
                if fb.button("Reset all", use_container_width=True):
                    st.session_state[state.FC_FLIPPED] = set()
                    st.rerun()

                cols = st.columns(2)
                for i, card in enumerate(fc_data):
                    flipped = i in fc_flipped
                    content = card.get("back" if flipped else "front", "")
                    with cols[i % 2]:
                        st.markdown(
                            f'<div class="{"fc flipped" if flipped else "fc"}">'
                            f'<div class="fc-label">{"BACK" if flipped else "FRONT"}</div>'
                            f'<div class="{"fc-back" if flipped else "fc-front"}">'
                            f'{content}</div></div>',
                            unsafe_allow_html=True)
                        if st.button("Flip", key=f"fc_{i}_{d_hash}",
                                     use_container_width=True):
                            if flipped:
                                st.session_state[state.FC_FLIPPED].discard(i)
                            else:
                                st.session_state[state.FC_FLIPPED].add(i)
                            st.rerun()

        # ── SIMPLIFY ──────────────────────────────────────────────────────────
        elif mode == "\u270f\ufe0f  Simplify":
            audience = st.radio("Explain as a:",
                                ["Complete beginner",
                                 "Curious teenager",
                                 "Busy professional"],
                                horizontal=True)

            if st.button("Simplify", use_container_width=True):
                ctx    = (
                    " ".join(chu.chunk_doc(doc_text)[:2])[:chu.MAX_CHARS]
                    if long_doc else doc_text[:chu.MAX_CHARS]
                )
                holder = st.empty()
                full   = ""
                for chunk in ai.stream_call(
                    prompts.simplify(ctx, audience), feature="simplify"
                ):
                    full += chunk
                    holder.markdown(
                        f'<div class="bk-card">'
                        f'<div class="bk-card-title">For: {audience}</div>'
                        f'<div class="bk-card-body">{full}</div></div>',
                        unsafe_allow_html=True)
                holder.empty()
                st.session_state[state.SIMPLIFY_TEXT] = full
                st.session_state[state.SIMPLIFY_AUD]  = audience
                # No st.rerun() - renders immediately below

            if st.session_state.get(state.SIMPLIFY_TEXT):
                st.markdown(
                    f'<div class="bk-card">'
                    f'<div class="bk-card-title">'
                    f'{st.session_state.get(state.SIMPLIFY_AUD, "Simplified")}</div>'
                    f'<div class="bk-card-body">'
                    f'{st.session_state[state.SIMPLIFY_TEXT]}</div></div>',
                    unsafe_allow_html=True)
                st.download_button(
                    "Export Explanation",
                    data=st.session_state[state.SIMPLIFY_TEXT],
                    file_name="bookiee_simplified.txt",
                    mime="text/plain")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='text-align:center;font-size:.75rem;color:#484f57'>"
    f"Bookiee AI v{APP_VERSION} \u00b7 Streamlit + Gemini 2.5 Flash \u00b7 "
    "<a href='https://github.com' style='color:#3fb950;text-decoration:none'>GitHub</a></p>",
    unsafe_allow_html=True)