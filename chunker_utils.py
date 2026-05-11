"""
chunker_utils.py - Document chunking and context retrieval.
Imported by both ai_engine.py and app.py.
"""

import re
import hashlib

CHUNK_WORDS   = 1600
OVERLAP_WORDS = 160
MAX_CHARS     = 8000


def make_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def doc_meta(text: str) -> dict:
    w = len(text.split())
    return {"words": w, "chars": len(text), "read_mins": max(1, round(w / 238))}


def is_long(text: str) -> bool:
    return len(text) > MAX_CHARS * 1.1


def chunk_doc(text: str, chunk_words: int = CHUNK_WORDS,
              overlap_words: int = OVERLAP_WORDS) -> list:
    """Sentence-aware chunking with overlap for context continuity."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur, wc = [], [], 0
    for sent in sentences:
        sw = len(sent.split())
        if wc + sw > chunk_words and cur:
            chunks.append(" ".join(cur))
            overlap, carried = [], 0
            for s in reversed(cur):
                n = len(s.split())
                if carried + n <= overlap_words:
                    overlap.insert(0, s); carried += n
                else:
                    break
            cur, wc = overlap, carried
        cur.append(sent); wc += sw
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def smart_chat_ctx(text: str, question: str) -> str:
    """
    Keyword-scored chunk retrieval for long documents.
    Returns the most relevant chunk plus neighbours.
    No embeddings needed - keyword overlap is effective for study content.
    """
    if not is_long(text):
        return text[:MAX_CHARS]
    chunks = chunk_doc(text)
    if len(chunks) == 1:
        return chunks[0][:MAX_CHARS]
    q_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    scored = sorted(
        enumerate(chunks),
        key=lambda ic: len(q_words & set(ic[1].lower().split())),
        reverse=True,
    )
    best = scored[0][0]
    return " ".join(chunks[max(0, best - 1): best + 2])[:MAX_CHARS]