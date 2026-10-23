"""Rerank: Gemini scores each candidate from 0 to 10 and keeps the best."""

from core import prompts
from core.gemini import ask


def rerank(question: str, candidates: list[dict], keep: int = 4, cutoff: int = 3) -> list[dict]:
    passages = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(candidates))
    reply = ask(prompts.RERANK.format(question=question, passages=passages), schema=prompts.RERANK_REPLY)
    scores = {s["n"]: s["score"] for s in reply}
    ranked = sorted(({**c, "rerank": scores.get(n, 0)} for n, c in enumerate(candidates)), key=lambda c: -c["rerank"])
    return [c for c in ranked[:keep] if c["rerank"] >= cutoff] or ranked[:1]
