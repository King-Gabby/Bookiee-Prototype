"""
prompts.py — Centralized prompt registry for Bookiee AI.

Every AI call in the app uses a function from here.
Benefits: versioning, testing, reuse, single place to tune quality.
"""

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_STUDY = (
    "You are Bookiee AI — a sharp, encouraging study assistant. "
    "Be precise, insightful, and educational. Never make things up."
)

SYSTEM_JSON = (
    "Return ONLY raw valid JSON. "
    "No markdown fences, no preamble, no explanation, no trailing text."
)


# ── Analysis prompts (streaming) ───────────────────────────────────────────────
def summary(context: str) -> str:
    return (
        "Write a clear, insightful summary in 3-4 well-structured paragraphs. "
        "Cover the main argument, key evidence, and key implications. "
        "Write for an intelligent adult reader:\n\n"
        f"{context}"
    )

def deep_dive(term: str, context: str) -> str:
    return (
        f"Give a richer 2-3 paragraph explanation of the concept '{term}' "
        f"as it appears in the document below. "
        f"Include a real-world example or analogy to make it concrete:\n\n"
        f"{context}"
    )

def simplify(context: str, audience: str = "Complete beginner") -> str:
    personas = {
        "Complete beginner": (
            "complete beginner with no prior knowledge. "
            "Use simple words, relatable analogies, and everyday examples. "
            "Avoid jargon entirely."
        ),
        "Curious teenager": (
            "curious 16-year-old. Be engaging and energetic. "
            "Use pop culture references where helpful. Keep it lively and direct."
        ),
        "Busy professional": (
            "busy professional who needs the core insight in under 3 minutes. "
            "Be dense, direct, and focus on actionable implications. "
            "Lead with the bottom line."
        ),
    }
    persona = personas.get(audience, personas["Complete beginner"])
    return (
        f"Explain the following text to a {persona}\n\n"
        f"{context}"
    )

def chat(context: str, question: str) -> str:
    return (
        "Answer the following question based only on the context provided below. "
        "If the answer is not in the context, say so clearly rather than guessing. "
        "Be specific and reference relevant parts of the text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


# ── JSON prompts (structured output) ──────────────────────────────────────────
def key_points(context: str) -> str:
    return (
        "Extract exactly 7 key ideas from the text below. "
        "Each idea should be a standalone insight under 15 words. "
        "Return a JSON array of strings.\n\n"
        f"{context}"
    )

def concepts(context: str) -> str:
    return (
        "Identify exactly 5 important concepts from the text below. "
        "Return a JSON array of objects. Each object must have exactly two fields: "
        "'term' (the concept name, 1-4 words) and "
        "'definition' (a clear one-sentence explanation).\n\n"
        f"{context}"
    )

def quiz(context: str, difficulty: str = "Medium", n: int = 5) -> str:
    hint = {
        "Easy":   "straightforward recall and comprehension",
        "Medium": "comprehension, application, and interpretation",
        "Hard":   "analysis, synthesis, evaluation, and critical thinking",
    }.get(difficulty, "comprehension and application")
    return (
        f"Create exactly {n} multiple-choice questions testing {hint} "
        f"of the text below. "
        f"Return a JSON array of objects. Each object must have exactly these fields: "
        f"'question' (the question string), "
        f"'options' (array of exactly 4 strings, each prefixed 'A. ', 'B. ', 'C. ', 'D. '), "
        f"'answer' (single letter: A, B, C, or D — just the letter, nothing else), "
        f"'explanation' (1-2 sentences explaining why the answer is correct).\n\n"
        f"{context}"
    )

def flashcards(context: str, n: int = 8) -> str:
    return (
        f"Create exactly {n} flashcards from the text below. "
        f"Return a JSON array of objects. Each object must have exactly two fields: "
        f"'front' (a concise term, concept, or question — under 10 words) and "
        f"'back' (a clear definition or answer — 1-2 sentences).\n\n"
        f"{context}"
    )

def follow_up(question: str, answer: str) -> str:
    return (
        "Based on the Q&A exchange below, suggest exactly 3 natural follow-up questions "
        "the user might want to ask next. Keep each question under 10 words. "
        "Return a JSON array of 3 strings.\n\n"
        f"Q: {question}\nA: {answer[:500]}"
    )


# ── Chunking prompts (for long documents) ─────────────────────────────────────
def chunk_summary(chunk: str, chunk_num: int, total_chunks: int) -> str:
    return (
        f"You are reading section {chunk_num} of {total_chunks} from a longer document. "
        f"Summarize ALL the key information from this section. "
        f"Be comprehensive — this summary will be used to understand the full document:\n\n"
        f"{chunk}"
    )

def synthesize_summary(section_summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return (
        "You have been given summaries of each section of a longer document. "
        "Write a single unified, coherent summary in 3-4 paragraphs "
        "that captures the main argument, key evidence, and implications "
        "of the complete document:\n\n"
        f"{combined}"
    )

def synthesize_key_points(section_summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return (
        "You have been given summaries of each section of a longer document. "
        "Extract exactly 7 key ideas that represent the most important insights "
        "across the ENTIRE document. Avoid redundancy. "
        "Return a JSON array of strings, each under 15 words.\n\n"
        f"{combined}"
    )

def synthesize_concepts(section_summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return (
        "You have been given summaries of each section of a longer document. "
        "Identify exactly 5 key concepts that appear across the complete document. "
        "Return a JSON array of objects with 'term' and 'definition' fields.\n\n"
        f"{combined}"
    )

def synthesize_quiz(section_summaries: list[str], difficulty: str, n: int) -> str:
    hint = {
        "Easy":   "recall and comprehension",
        "Medium": "comprehension and application",
        "Hard":   "analysis and critical thinking",
    }.get(difficulty, "comprehension")
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return (
        f"You have been given summaries of each section of a longer document. "
        f"Create exactly {n} multiple-choice questions testing {hint} "
        f"of the complete document. Draw questions from multiple sections. "
        f"Return a JSON array of objects with: "
        f"'question', 'options' (4 strings prefixed A. B. C. D.), "
        f"'answer' (single letter), 'explanation' (1-2 sentences).\n\n"
        f"{combined}"
    )

def synthesize_flashcards(section_summaries: list[str], n: int) -> str:
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return (
        f"You have been given summaries of each section of a longer document. "
        f"Create exactly {n} flashcards covering key terms and concepts "
        f"from across the ENTIRE document. "
        f"Return a JSON array of objects with 'front' and 'back' fields.\n\n"
        f"{combined}"
    )

def find_relevant_chunk(chunks_summary: str, question: str) -> str:
    return (
        "Given the following document section summaries and a user question, "
        "identify which section number (1, 2, 3, etc.) is most relevant to answering the question. "
        "Return a JSON object with a single field: 'section' (integer).\n\n"
        f"Summaries:\n{chunks_summary}\n\n"
        f"Question: {question}"
    )