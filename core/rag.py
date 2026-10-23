"""The four kinds of RAG in this workshop, built from small steps you can call one at a time.

    naive        fixed chunks, top 3 by similarity, answer
    text         careful chunks, metadata filters, reranking, answers that cite page numbers
    multimodal   text plus pictures, recordings and videos, searched by meaning and shown to Gemini
    graph        a vector search finds where to start, the knowledge graph connects the rest
"""

from functools import cache

import numpy as np
from google.genai import types

from core.answer import answer, shown
from core.gemini import ask, embed
from core.graph import Graph
from core.store import Store


@cache
def store(name: str) -> Store:
    return Store(name)


@cache
def graph() -> Graph:
    return Graph()


# ------------------------------------------------------------------ naive

NAIVE_PROMPT = "Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {question}"


def naive(question: str) -> dict:
    """The tutorial version: chop text every 500 characters, take the 3 most similar, answer."""
    found = store("naive").search(embed(question), k=3)
    context = "\n\n".join(chunk["text"] for chunk in found)
    return {"answer": ask(NAIVE_PROMPT.format(context=context, question=question)), "sources": [shown(c) for c in found]}


# ------------------------------------------------------------------ text, done right

def find_text(question: str, k: int = 12, kinds=None, after=None, latest_only=False) -> list[dict]:
    """Semantic chunks, with the metadata filter applied before ranking."""
    return store("text").search(embed(question), k=k, kinds=kinds, after=after, latest_only=latest_only)


SCORES = types.Schema(type="ARRAY", items=types.Schema(
    type="OBJECT",
    properties={"n": types.Schema(type="INTEGER"), "score": types.Schema(type="INTEGER")},
    required=["n", "score"],
))

RERANK_PROMPT = """For each numbered passage, score from 0 to 10 how well it answers the question.
10 = it contains the answer. 0 = it is about something else. Similar words are not enough.

Question: {question}

{passages}"""


def rerank(question: str, candidates: list[dict], keep: int = 4, cutoff: int = 3) -> list[dict]:
    """Similarity finds passages that look like the question. A reranker reads each one and asks:
    does this actually answer it? One Gemini call scores all candidates; weak ones are dropped."""
    passages = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(candidates))
    scores = {s["n"]: s["score"] for s in ask(RERANK_PROMPT.format(question=question, passages=passages), schema=SCORES)}
    ranked = [{**candidates[n], "rerank": scores.get(n, 0)} for n in sorted(range(len(candidates)), key=lambda n: -scores.get(n, 0))]
    return [c for c in ranked[:keep] if c["rerank"] >= cutoff] or ranked[:1]


def text(question: str, kinds=None, after=None, latest_only=True) -> dict:
    candidates = find_text(question, kinds=kinds, after=after, latest_only=latest_only)
    return answer(question, rerank(question, candidates))


# ------------------------------------------------------------------ multimodal

def find_media(question: str, k: int = 2) -> list[dict]:
    """Each picture, recording and video is scored twice, as itself and by its description, and the
    better score counts. A sentence compared with a recording is not on the same scale as a sentence
    compared with a sentence (the "modality gap"), so the description keeps the ranking honest."""
    vector = embed(question)
    itself, described = store("media"), store("media-descriptions")
    scores = np.maximum(itself.vectors @ vector, described.vectors @ vector)
    return [{**itself.records[i], "score": round(float(scores[i]), 3)} for i in np.argsort(-scores)[:k]]


def multimodal(question: str, kinds=None, after=None, latest_only=True) -> dict:
    passages = rerank(question, find_text(question, kinds=kinds, after=after, latest_only=latest_only), keep=3)
    return answer(question, passages + find_media(question))


# ------------------------------------------------------------------ graph

def graph_facts(question: str) -> tuple[set[str], list[dict]]:
    """Start from the names in the question and in the best two chunks, then walk three hops."""
    entry = find_text(question, k=2, latest_only=True)
    start = graph().named_in(question) | graph().from_chunks({c["id"] for c in entry})
    # The same metadata filter as text RAG: a fact read from a replaced document does not count.
    current = {r["id"] for name in ("text", "media") for r in store(name).records if store(name).passes(r, None, None, True)}
    return start, [fact for fact in graph().walk(start) if fact["source"] in current]


def graph_rag(question: str) -> dict:
    start, facts = graph_facts(question)
    by_id = {r["id"]: r for r in store("text").records + store("media").records}
    sources = list({f["source"]: by_id[f["source"]] for f in facts if f["source"] in by_id}.values())
    number = {source["id"]: n for n, source in enumerate(sources, start=1)}
    facts = [{**f, "n": number.get(f["source"], 0)} for f in facts]
    result = answer(question, sources, facts=facts)
    result["graph"] = {"start": sorted(start), "facts": facts}
    return result


def record(ref: str) -> dict:
    """Any chunk or media item by its id: n for naive chunks, t for text chunks, m for media."""
    return next(r for name in ("naive", "text", "media") for r in store(name).records if r["id"] == ref)
