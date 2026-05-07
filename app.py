import streamlit as st
from google import genai
import json
import pdfplumber
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bookiee AI - Prototype",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Header */
.bk-header { text-align: center; padding: 2rem 0 1rem; }
.bk-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #2ea043, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .25rem;
}
.bk-header p { color: #8b949e; font-size: .95rem; margin-top: 0; }

/* Cards */
.bk-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.bk-card-title {
    font-size: .75rem;
    font-weight: 600;
    color: #3fb950;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: .75rem;
}
.bk-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: .8rem;
    color: #c9d1d9;
    margin: 3px;
}
.bk-concept {
    border-left: 2px solid #3fb950;
    padding: 8px 14px;
    margin-bottom: 8px;
    background: #0d1117;
    border-radius: 0 6px 6px 0;
}
.bk-concept strong { color: #e6edf3; font-size: .85rem; }
.bk-concept span   { color: #8b949e;  font-size: .8rem;  }

/* Chat bubbles */
.chat-user {
    background: #2ea043; color: #fff;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px; margin: 6px 0;
    max-width: 80%; margin-left: auto;
    font-size: .875rem;
}
.chat-bot {
    background: #21262d; color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 2px 12px 12px 12px;
    padding: 10px 14px; margin: 6px 0;
    max-width: 85%;
    font-size: .875rem; line-height: 1.65;
}

/* Quiz */
.quiz-q {
    background: #0d1117; border: 1px solid #30363d;
    border-radius: 8px; padding: 1rem; margin-bottom: .75rem;
}
.quiz-q h4 { color: #e6edf3; font-size: .875rem; margin-bottom: .5rem; }
.quiz-correct { color: #3fb950 !important; font-weight: 600; }
.quiz-wrong   { color: #f85149 !important; }

/* Flashcards */
.fc {
    background: #0d1117; border: 1.5px solid #30363d;
    border-radius: 10px; padding: 1rem;
    text-align: center; min-height: 100px;
    display: flex; align-items: center; justify-content: center;
}
.fc-term { font-size: .9rem; font-weight: 600; color: #e6edf3; }
.fc-def  { font-size: .8rem; color: #8b949e; line-height: 1.6; }

/* Sidebar */
[data-testid="stSidebar"] { background: #161b22; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #3fb950; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: .5rem 1.5rem !important;
}
.stButton > button:hover { opacity: .9 !important; transform: translateY(-1px); }

div[data-baseweb="tab-list"] button { font-family: 'DM Sans', sans-serif; }
</style>
""", unsafe_allow_html=True)

# I am temporarily using Gemini API for the sake of prototype.

# ── Gemini client ───────────────────────────────────────────────────────────
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("⚠️ Add GEMINI_API_KEY in Settings → Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)


def call_gemini(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Single wrapper for all Gemini calls (Claude-style parity)."""

    client = get_client()

    sys_prompt = system or (
        "Return ONLY valid JSON, no markdown fences, no explanation."
        if json_mode else
        "You are Bookiee AI, a sharp and friendly study assistant."
    )

 
    full_prompt = f"{sys_prompt}\n\n{prompt}"

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=full_prompt
        )

        # Gemini can return multiple parts , gotta handle safely
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "".join(
            part.text for part in response.candidates[0].content.parts
            if hasattr(part, "text")
        ).strip()

    except Exception as e:
        return f"Error: {e}"


def call_gemini_json(prompt: str):
    """Call Gemini and parse JSON response (Claude-style)."""
    raw = call_gemini(prompt, json_mode=True)

    try:
        cleaned = (
            raw.replace("```json", "")
               .replace("```", "")
               .strip()
        )

        # Gemini sometimes adds text before/after JSON
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        if start != -1 and end != -1:
            cleaned = cleaned[start:end]

        return json.loads(cleaned)

    except json.JSONDecodeError:
        return None

# ── PDF / text extraction ──────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    """Extract plain text from .txt, .md, or .pdf uploads."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages[:30]
            )
    # txt / md
    return uploaded_file.read().decode("utf-8", errors="ignore")


# ── Session state defaults ─────────────────────────────────────────────────────
for key, val in {
    "doc_text": "",
    "analysis": None,
    "chat_history": [],
    "quiz_data": [],
    "quiz_answers": {},
    "fc_data": [],
    "fc_flipped": set(),
    "simplify_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — document input
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📚 Bookiee AI")
    st.markdown("---")
    st.markdown("**Upload a document**")

    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["txt", "pdf", "md"],
        label_visibility="collapsed",
    )
    if uploaded:
        with st.spinner("Reading file…"):
            st.session_state.doc_text = extract_text(uploaded)
        st.success(f"✅ {uploaded.name} loaded ({len(st.session_state.doc_text):,} chars)")

    st.markdown("**or paste text**")
    pasted = st.text_area(
        "Paste content here",
        height=180,
        placeholder="Paste an article, chapter, notes…",
        label_visibility="collapsed",
    )
    if pasted:
        st.session_state.doc_text = pasted

    has_doc = len(st.session_state.doc_text.strip()) > 100

    if has_doc:
        st.info(f"📄 {len(st.session_state.doc_text):,} characters ready")
        if st.button("🗑️ Clear document"):
            for k in ["doc_text", "analysis", "chat_history",
                      "quiz_data", "quiz_answers", "fc_data",
                      "fc_flipped", "simplify_text"]:
                st.session_state[k] = [] if k in ("chat_history","quiz_data","fc_data","fc_flipped") else (
                    {} if k == "quiz_answers" else "")
            st.rerun()
    else:
        st.caption("Add a document to unlock all features.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#484f57'>Bookiee-Prototype · "
        "[Source on GitHub](https://github.com) · "
        "[Deploy on Streamlit](https://streamlit.io)</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — header + tabs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bk-header">
  <h1>📚 Bookiee AI - Prototype</h1>
  <p>Upload → Analyze → Study smarter. Powered by CelesTium-AI.</p>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_chat, tab_study = st.tabs(
    ["📄  Analyze Document", "💬  Ask Questions", "🧠  Study Mode"]
)
ctx = st.session_state.doc_text[:5000]   # token budget for API calls


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    if not has_doc:
        st.info("⬅️  Upload a document or paste text in the sidebar to begin.")
    else:
        if st.button("✨ Analyze Document", use_container_width=True):
            with st.spinner("Running analysis — fetching summary, key points & concepts…"):
                summary    = call_gemini(f"Summarize this text in 3-4 clear paragraphs:\n\n{ctx}")
                key_points = call_gemini_json(
                    f"Extract exactly 6 key ideas. Return a JSON array of strings (≤15 words each):\n\n{ctx}"
                )
                concepts   = call_gemini_json(
                    f"Identify 4-5 key concepts. Return JSON array of {{term, definition}} objects:\n\n{ctx}"
                )
                st.session_state.analysis = {
                    "summary": summary,
                    "key_points": key_points or [],
                    "concepts": concepts or [],
                }

        if st.session_state.analysis:
            a = st.session_state.analysis

            # Summary
            st.markdown('<div class="bk-card"><div class="bk-card-title">📝 Summary</div>', unsafe_allow_html=True)
            st.write(a["summary"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Key Points
            st.markdown('<div class="bk-card"><div class="bk-card-title">🔑 Key Points</div>', unsafe_allow_html=True)
            pills = " ".join(f'<span class="bk-pill">• {p}</span>' for p in a["key_points"])
            st.markdown(pills, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Concepts
            st.markdown('<div class="bk-card"><div class="bk-card-title">💡 Important Concepts</div>', unsafe_allow_html=True)
            for c in a["concepts"]:
                st.markdown(
                    f'<div class="bk-concept"><strong>{c.get("term","")}</strong><br>'
                    f'<span>{c.get("definition","")}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not has_doc:
        st.info("⬅️  Add a document first, then ask anything about it.")
    else:
        st.caption("Ask anything about your document.")

        # Render history
        for msg in st.session_state.chat_history:
            css = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(
                f'<div class="{css}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        # Quick-start prompts
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            c1, c2, c3 = st.columns(3)
            prompts = [
                "Summarise chapter 1 simply",
                "What are the key arguments?",
                "Give me 3 takeaways",
            ]
            for col, p in zip([c1, c2, c3], prompts):
                if col.button(p, use_container_width=True):
                    st.session_state._quick_prompt = p
                    st.rerun()

        user_input = st.chat_input("Ask a question about your document…")
        if hasattr(st.session_state, "_quick_prompt"):
            user_input = st.session_state._quick_prompt
            del st.session_state._quick_prompt

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                answer = call_gemini(
                    f"Answer this question based only on the context below.\n\n"
                    f"Context:\n{ctx}\n\nQuestion: {user_input}"
                )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STUDY MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab_study:
    if not has_doc:
        st.info("⬅️  Add a document to generate study materials.")
    else:
        mode = st.radio(
            "Choose study format",
            ["📝 Quiz", "🃏 Flashcards", "✏️ Simplify"],
            horizontal=True,
        )

        if st.button("🚀 Generate", use_container_width=True):
            if mode == "📝 Quiz":
                with st.spinner("Building your quiz…"):
                    data = call_gemini_json(
                        f"Create 5 multiple-choice questions. Return JSON array of "
                        f"{{question, options: [A.x, B.x, C.x, D.x], answer: letter, explanation}} objects:\n\n{ctx}"
                    )
                    st.session_state.quiz_data    = data or []
                    st.session_state.quiz_answers = {}

            elif mode == "🃏 Flashcards":
                with st.spinner("Creating flashcards…"):
                    data = call_gemini_json(
                        f"Create 8 flashcards. Return JSON array of {{front, back}} objects:\n\n{ctx}"
                    )
                    st.session_state.fc_data    = data or []
                    st.session_state.fc_flipped = set()

            else:
                with st.spinner("Simplifying…"):
                    st.session_state.simplify_text = call_gemini(
                        f"Explain this for a complete beginner. Use plain language, "
                        f"analogies, and examples. Be warm and clear:\n\n{ctx}"
                    )

        # ── Render Quiz ──────────────────────────────────────────────────────
        if mode == "📝 Quiz" and st.session_state.quiz_data:
            correct = sum(
                1 for i, q in enumerate(st.session_state.quiz_data)
                if st.session_state.quiz_answers.get(i) == q.get("answer")
            )
            answered = len(st.session_state.quiz_answers)
            total    = len(st.session_state.quiz_data)
            st.markdown(f"**Score: {correct}/{total}** &nbsp;·&nbsp; {answered} answered")

            for i, q in enumerate(st.session_state.quiz_data):
                with st.expander(f"Q{i+1}. {q.get('question','')}", expanded=True):
                    chosen = st.radio(
                        "Choose an answer",
                        q.get("options", []),
                        key=f"quiz_{i}",
                        label_visibility="collapsed",
                    )
                    if st.button("Submit", key=f"submit_{i}"):
                        letter = chosen[0] if chosen else ""
                        st.session_state.quiz_answers[i] = letter
                        st.rerun()

                    if i in st.session_state.quiz_answers:
                        given   = st.session_state.quiz_answers[i]
                        correct_ans = q.get("answer", "")
                        if given == correct_ans:
                            st.success(f"✅ Correct! ({correct_ans})")
                        else:
                            st.error(f"❌ You chose {given} — correct answer is **{correct_ans}**")
                        st.caption(q.get("explanation", ""))

        # ── Render Flashcards ────────────────────────────────────────────────
        elif mode == "🃏 Flashcards" and st.session_state.fc_data:
            st.caption("Click **Flip** to reveal the answer.")
            cols = st.columns(2)
            for i, card in enumerate(st.session_state.fc_data):
                with cols[i % 2]:
                    flipped = i in st.session_state.fc_flipped
                    label   = "back" if flipped else "front"
                    content = card.get("back" if flipped else "front", "")
                    css     = "fc-def" if flipped else "fc-term"
                    color   = "#3fb950" if flipped else "#e6edf3"
                    st.markdown(
                        f'<div class="fc" style="border-color:{"#3fb950" if flipped else "#30363d"}">'
                        f'<div><p style="font-size:.65rem;color:#484f57;margin:0">{label}</p>'
                        f'<p class="{css}" style="color:{color};margin:.25rem 0 0">{content}</p></div></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Flip 🔄", key=f"fc_{i}", use_container_width=True):
                        if flipped:
                            st.session_state.fc_flipped.discard(i)
                        else:
                            st.session_state.fc_flipped.add(i)
                        st.rerun()

        # ── Render Simplify ──────────────────────────────────────────────────
        elif mode == "✏️ Simplify" and st.session_state.simplify_text:
            st.markdown('<div class="bk-card"><div class="bk-card-title">✏️ Simplified Explanation</div>', unsafe_allow_html=True)
            st.write(st.session_state.simplify_text)
            st.markdown('</div>', unsafe_allow_html=True)