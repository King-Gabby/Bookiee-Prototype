"""
state.py - Centralized session state for Bookiee AI.
All keys namespaced under "bk_" to prevent Streamlit collisions.
No other module touches st.session_state directly.
"""

import streamlit as st
from datetime import datetime

_NS = "bk_"
def _k(n): return _NS + n

# Document
DOC_TEXT      = _k("doc_text")
DOC_SRC_ID    = _k("doc_source_id")
DOC_FILENAME  = _k("doc_filename")
IS_DEMO       = _k("is_demo")

# Generated content — all reset when document changes
ANALYSIS      = _k("analysis")
DEEP_DIVES    = _k("deep_dives")
CHAT_HISTORY  = _k("chat_history")
LAST_FOLLOWUPS= _k("last_followups")
QUIZ_DATA     = _k("quiz_data")
QUIZ_ANSWERS  = _k("quiz_answers")
FC_DATA       = _k("fc_data")
FC_FLIPPED    = _k("fc_flipped")
SIMPLIFY_TEXT = _k("simplify_text")
SIMPLIFY_AUD  = _k("simplify_aud")

# UI state — persists across document changes
PENDING_Q     = _k("pending_q")

# Session analytics — never reset
DOCS_ANALYZED = _k("docs_analyzed")
QS_ASKED      = _k("qs_asked")
QUIZZES_TAKEN = _k("quizzes_taken")
SESSION_START = _k("session_start")

# AI observability
LAST_AI_TS    = _k("last_ai_ts")
AI_CALL_COUNT = _k("ai_call_count")
AI_LOG        = _k("ai_log")

# Mutable types — instantiated fresh per session (never shared references)
_MUTABLE = {
    CHAT_HISTORY:   list,
    LAST_FOLLOWUPS: list,
    QUIZ_DATA:      list,
    QUIZ_ANSWERS:   dict,
    FC_DATA:        list,
    FC_FLIPPED:     set,
    DEEP_DIVES:     dict,
    AI_LOG:         list,
}

_SCALAR = {
    DOC_TEXT:      "",
    DOC_SRC_ID:    "",
    DOC_FILENAME:  "",
    IS_DEMO:       False,
    ANALYSIS:      None,
    SIMPLIFY_TEXT: "",
    SIMPLIFY_AUD:  "",
    PENDING_Q:     "",
    DOCS_ANALYZED: 0,
    QS_ASKED:      0,
    QUIZZES_TAKEN: 0,
    LAST_AI_TS:    0.0,
    AI_CALL_COUNT: 0,
}


def init() -> None:
    """
    Initialize all session state. Safe on every rerun -
    only sets keys that do not exist yet, never overwrites existing state.
    """
    for key, factory in _MUTABLE.items():
        if key not in st.session_state:
            st.session_state[key] = factory()
    for key, default in _SCALAR.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if SESSION_START not in st.session_state:
        st.session_state[SESSION_START] = datetime.now().strftime("%H:%M")


def has_doc() -> bool:
    return len(st.session_state.get(DOC_TEXT, "").strip()) > 80


def set_document(text: str, source_id: str,
                 filename: str = "", is_demo: bool = False) -> bool:
    """
    THE core fix for the upload lifecycle / refresh bug.

    Problem: st.file_uploader() returns its file object on EVERY Streamlit rerun,
    not only when a new file is uploaded. The old code called reset_doc_derived()
    unconditionally inside `if uploaded:`, so every button click, quiz submission,
    or tab switch triggered the upload handler and wiped all generated content.

    Fix: compare source_id. If unchanged -> no-op, preserves everything.
    Only a genuinely new document triggers a state reset.

    Returns True if document changed, False if same document (no-op).
    """
    if source_id == st.session_state.get(DOC_SRC_ID, ""):
        return False
    st.session_state[DOC_TEXT]     = text
    st.session_state[DOC_SRC_ID]   = source_id
    st.session_state[DOC_FILENAME] = filename
    st.session_state[IS_DEMO]      = is_demo
    _reset_doc_derived()
    return True


def _reset_doc_derived() -> None:
    """Reset content derived from the document. Never touches session counters."""
    st.session_state[ANALYSIS]       = None
    st.session_state[CHAT_HISTORY]   = []
    st.session_state[LAST_FOLLOWUPS] = []
    st.session_state[QUIZ_DATA]      = []
    st.session_state[QUIZ_ANSWERS]   = {}
    st.session_state[FC_DATA]        = []
    st.session_state[FC_FLIPPED]     = set()
    st.session_state[SIMPLIFY_TEXT]  = ""
    st.session_state[SIMPLIFY_AUD]   = ""
    st.session_state[DEEP_DIVES]     = {}