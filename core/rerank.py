"""Reranking: similarity finds passages that look like the question; a reranker reads each one and asks
whether it actually answers it. One Gemini call scores every candidate, and the weak ones are dropped.

Search engines use a cross-encoder model for this, such as the Vertex AI ranking API. An LLM with a
scoring prompt does the same job and shows you exactly what it is judging.
"""

from core import prompts
from core.gemini import ask


def rerank(question: str, candidates: list[dict], keep: int = 4, cutoff: int = 3) -> list[dict]:
    passages = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(candidates))
    reply = ask(prompts.RERANK.format(question=question, passages=passages), schema=prompts.RERANK_REPLY)
    scores = {s["n"]: s["score"] for s in reply}
    ranked = sorted(({**c, "rerank": scores.get(n, 0)} for n, c in enumerate(candidates)), key=lambda c: -c["rerank"])
    return [c for c in ranked[:keep] if c["rerank"] >= cutoff] or ranked[:1]
