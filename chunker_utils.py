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


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT PREPROCESSING — strip metadata before AI generation
# ══════════════════════════════════════════════════════════════════════════════

# Section headers that signal non-educational content.
# Match is case-insensitive against the stripped first line of each paragraph.
_METADATA_HEADERS = {
    "about the author", "about the authors", "about the writer",
    "author biography", "author bio", "author note", "about the editor",
    "acknowledgements", "acknowledgments", "acknowledgement",
    "preface", "foreword", "prologue",
    "table of contents", "contents",
    "copyright", "all rights reserved",
    "references", "bibliography", "works cited", "citations",
    "further reading", "suggested reading",
    "index",
    "publisher", "published by", "colophon",
    "dedication", "epigraph",
    "isbn", "issn",
    "permissions", "permissions acknowledgements",
    "editorial note", "series editor note",
    "note on translation", "translator note",
    "list of figures", "list of tables", "list of illustrations",
    "glossary",
}

# Inline patterns that mark a paragraph as metadata regardless of heading
_METADATA_LINE_PATTERNS = [
    r'^\s*isbn[\s\-:]',
    r'^\s*issn[\s\-:]',
    r'^\s*(copyright|\u00a9|\(c\))\s+\d{4}',
    r'^\s*all\s+rights\s+reserved',
    r'^\s*printed\s+in\b',
    r'^\s*published\s+by\b',
    r'^\s*\d{4}\s+\w[\w\s]+\s+(press|publishing|publishers)\b',
    r'^\s*first\s+published\b',
    r'^\s*typeset\s+by\b',
]

# Words that boost educational importance score
_EDUCATIONAL_SIGNALS = [
    "because", "therefore", "thus", "however", "although", "whereas",
    "defines", "means", "refers to", "is defined as", "is known as",
    "explains", "demonstrates", "shows", "illustrates", "describes",
    "for example", "such as", "in particular", "specifically",
    "furthermore", "moreover", "in addition", "consequently",
    "the process", "the method", "the principle", "the concept",
    "in order to", "as a result", "it follows that",
]

# Words that reduce educational importance score (metadata signals)
_METADATA_SIGNALS = [
    "isbn", "issn", "copyright", "all rights reserved",
    "published by", "et al.", "ibid", "op. cit.",
    "acknowledgement", "acknowledgment",
    "preface", "foreword", "table of contents",
    "bibliography", "references", "index",
]


def _is_metadata_block(para: str) -> bool:
    """
    Return True if this paragraph block is non-educational metadata.
    Uses two-pass detection:
    1. First-line header match against _METADATA_HEADERS
    2. Full-text pattern match against _METADATA_LINE_PATTERNS
    """
    lines = para.strip().split("\n")
    first = lines[0].strip().lower().rstrip(":").rstrip(".").strip()

    # Pass 1: exact header match
    if first in _METADATA_HEADERS:
        return True

    # Partial prefix match for common patterns
    for header in _METADATA_HEADERS:
        if first.startswith(header + " ") or first.startswith(header + "\t"):
            return True

    # Pass 2: pattern match on first 3 lines
    sample = "\n".join(lines[:3])
    for pat in _METADATA_LINE_PATTERNS:
        if re.search(pat, sample, re.IGNORECASE):
            return True

    return False


def _score_paragraph(para: str) -> float:
    """
    Score a paragraph's educational importance (0.0 to 1.0).
    Used for content-importance filtering in large documents.

    Heuristics:
    - Long sentences  = more informational
    - Educational signals = higher score
    - Metadata signals = lower score
    - Very short blocks = likely headings (neutral low score)
    """
    words = para.split()
    word_count = len(words)
    if word_count < 8:
        return 0.25

    text_lower = para.lower()
    score = 0.45  # baseline

    # Sentence density (longer sentences = more informational)
    sentence_count = max(1, para.count(".") + para.count("!") + para.count("?"))
    avg_len = word_count / sentence_count
    if avg_len > 18:
        score += 0.15
    elif avg_len > 12:
        score += 0.08

    # Educational signals
    for signal in _EDUCATIONAL_SIGNALS:
        if signal in text_lower:
            score += 0.04
            if score > 0.95:
                break

    # Metadata signals
    for signal in _METADATA_SIGNALS:
        if signal in text_lower:
            score -= 0.12

    # Word count bonus (dense paragraphs are more educational)
    if word_count > 80:
        score += 0.1
    elif word_count > 40:
        score += 0.05

    return max(0.0, min(1.0, score))


def clean_core_text(text: str, importance_threshold: float = 0.35) -> str:
    """
    LAYER 1 preprocessing: remove metadata sections and low-importance content.

    Three-stage pipeline:
    1. Metadata header filter: remove blocks whose first line matches known
       non-educational headers (About the Author, Acknowledgements, etc.)
    2. Pattern filter: remove blocks containing copyright / ISBN markers
    3. Importance filter: remove low-scoring paragraphs (below threshold)
       This catches misc. metadata that didn't match explicit patterns.

    Returns clean educational content only.
    The original text is never modified — a new string is returned.

    Usage:
        clean = clean_core_text(raw_pdf_text)
        # Use `clean` for all AI generation calls
    """
    # Split into paragraph-like blocks (double newline boundary)
    # Fall back to single newlines if the document has no double newlines
    if "\n\n" in text:
        blocks = re.split(r"\n{2,}", text)
    else:
        # Single-newline documents: split on long lines to avoid one big block
        blocks = re.split(r"\n(?=\S)", text)

    kept = []
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue

        # Stage 1 + 2: explicit metadata detection
        if _is_metadata_block(stripped):
            continue

        # Stage 3: importance threshold filter
        # Only applies to mid-length blocks (short = likely heading, keep it)
        word_count = len(stripped.split())
        if word_count > 20:
            score = _score_paragraph(stripped)
            if score < importance_threshold:
                continue

        kept.append(stripped)

    if not kept:
        # Safety net: if everything was filtered (e.g. very short document),
        # return the original rather than an empty string
        return text

    return "\n\n".join(kept)


def _filter_question_metadata(questions: list) -> list:
    """
    LAYER 3 post-generation filter: reject questions about document metadata.
    Called after Gemini returns quiz questions.

    Removes questions whose text contains known metadata trigger words.
    On rejection, the caller should regenerate (or skip) the question.
    """
    _TRIGGERS = {
        "author", "preface", "acknowledgement", "acknowledgment",
        "table of contents", "published by", "foreword",
        "bibliography", "isbn", "copyright", "publisher",
        "chapter list", "about the", "dedication",
        "who wrote", "who is the author",
    }
    clean = []
    for q in questions:
        text = (q.get("question", "") + " " +
                " ".join(q.get("options", []))).lower()
        if not any(t in text for t in _TRIGGERS):
            clean.append(q)
    return clean