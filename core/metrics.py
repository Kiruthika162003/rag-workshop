"""RAG metrics: retrieval (hit@k, precision@k, recall@k, MRR, nDCG) and judged (faithfulness,
answer relevancy, context precision, context recall)."""

import math
from concurrent.futures import ThreadPoolExecutor

from core import prompts
from core.embeddings import embed_passages
from core.gemini import ask, part
from core.store import INDEX


# ---------------------------------------------------------------- retrieval: a ranked list of True / False

def hit_at(relevant: list[bool], k: int) -> float:
    return float(any(relevant[:k]))


def precision_at(relevant: list[bool], k: int) -> float:
    return sum(relevant[:k]) / k


def recall_at(relevant: list[bool], k: int, total: int) -> float:
    return sum(relevant[:k]) / total if total else 0.0


def reciprocal_rank(relevant: list[bool]) -> float:
    return next((1 / place for place, hit in enumerate(relevant, start=1) if hit), 0.0)


def ndcg_at(relevant: list[bool], k: int, total: int) -> float:
    """Each relevant chunk is worth 1 / log2(place + 1); divided by the best possible ordering."""
    gained = sum(1 / math.log2(place + 1) for place, hit in enumerate(relevant[:k], start=1) if hit)
    best = sum(1 / math.log2(place + 1) for place in range(1, min(k, total) + 1))
    return gained / best if best else 0.0


# ---------------------------------------------------------------- the answer: a model as the judge

def evidence(sources: list[dict]) -> list:
    """The sources as the judge sees them: numbered, text as text, figures as pictures."""
    parts = []
    for n, source in enumerate(sources, start=1):
        if source["type"] == "picture":
            parts += [f"[{n}] a figure or table from page {source['page']}:", part((INDEX / source["media"]).read_bytes(), source["mime"])]
        else:
            parts.append(f"[{n}] {source['text']}")
    return parts


def faithfulness(answer: str, sources: list[dict]) -> dict:
    claims = ask([prompts.FAITHFUL, "Sources:", *evidence(sources), f"Answer: {answer}"], schema=prompts.FAITHFUL_REPLY)
    return {"score": sum(c["supported"] for c in claims) / len(claims) if claims else 0.0, "claims": claims}


def answer_relevancy(question: str, answer: str, n: int = 3) -> dict:
    """Questions the answer would answer, compared with the real one. "Could not find" scores 0."""
    reply = ask([prompts.ASKED.format(n=n), f"Answer: {answer}"], schema=prompts.ASKED_REPLY)
    if reply["noncommittal"] or not reply["questions"]:
        return {"score": 0.0, "questions": reply["questions"]}
    asked, *written = embed_passages([question, *reply["questions"]])
    return {"score": max(0.0, float(sum(asked @ w for w in written) / len(written))), "questions": reply["questions"]}


def context_precision(question: str, sources: list[dict], reference: str | None = None) -> dict:
    """Average precision: useful sources ranked first score 1."""
    task = f"Question: {question}" + (f"\nReference answer: {reference}" if reference else "")
    verdicts = {v["n"]: v["useful"] for v in ask([prompts.USEFUL, "Sources:", *evidence(sources), task], schema=prompts.USEFUL_REPLY)}
    useful = [bool(verdicts.get(n)) for n in range(1, len(sources) + 1)]
    score = sum(precision_at(useful, k) for k, hit in enumerate(useful, start=1) if hit) / sum(useful) if any(useful) else 0.0
    return {"score": score, "useful": useful}


def context_recall(reference: str, sources: list[dict]) -> dict:
    statements = ask([prompts.ATTRIBUTED, "Sources:", *evidence(sources), f"Reference answer: {reference}"],
                     schema=prompts.ATTRIBUTED_REPLY)
    return {"score": sum(s["found"] for s in statements) / len(statements) if statements else 0.0, "statements": statements}


def judge(question: str, answer: str, sources: list[dict], reference: str | None = None) -> dict:
    """All four at once, in parallel: one Gemini call each."""
    work = {"faithfulness": lambda: faithfulness(answer, sources),
            "answer_relevancy": lambda: answer_relevancy(question, answer),
            "context_precision": lambda: context_precision(question, sources, reference)}
    if reference:
        work["context_recall"] = lambda: context_recall(reference, sources)
    with ThreadPoolExecutor(len(work)) as pool:
        running = {name: pool.submit(job) for name, job in work.items()}
        return {name: job.result() for name, job in running.items()}
