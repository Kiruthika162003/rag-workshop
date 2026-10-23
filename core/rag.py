"""Every kind of RAG, a few lines each, built from the other files."""

from core import prompts
from core.answer import answer, shown
from core.gemini import ask
from core.graph import graph
from core.query import rewrites
from core.rerank import rerank
from core.search import by_meaning, fuse, hybrid, pictures, with_neighbours
from core.store import store


def naive(question: str) -> dict:
    """The tutorial version: chop text every 500 characters, take the 3 most similar, answer."""
    found = by_meaning(question, k=3, index="naive")
    context = "\n\n".join(chunk["text"] for chunk in found)
    return {"answer": ask(prompts.NAIVE.format(context=context, question=question)), "sources": [shown(c) for c in found]}


def text(question: str, section=None, pages=None) -> dict:
    best = rerank(question, hybrid(question, section=section, pages=pages))
    return answer(question, with_neighbours(best))


def multimodal(question: str, section=None, pages=None) -> dict:
    passages = rerank(question, hybrid(question, section=section, pages=pages), keep=3)
    return answer(question, with_neighbours(passages) + pictures(question))


def graph_facts(question: str) -> tuple[set[str], list[dict]]:
    """Start from the names in the question and in the best two chunks, then walk three hops."""
    entry = hybrid(question, k=2)
    start = graph().named_in(question) | graph().from_chunks({c["id"] for c in entry})
    return start, graph().walk(start)


def graph_rag(question: str) -> dict:
    start, facts = graph_facts(question)
    by_id = {r["id"]: r for r in store("text").records + store("pictures").records}
    sources = list({f["source"]: by_id[f["source"]] for f in facts if f["source"] in by_id}.values())
    number = {source["id"]: n for n, source in enumerate(sources, start=1)}
    facts = [{**f, "n": number.get(f["source"], 0)} for f in facts]
    result = answer(question, sources, facts=facts)
    result["graph"] = {"start": sorted(start), "facts": facts}
    return result


GOOD_ENOUGH = 7


def corrective(question: str) -> dict:
    """Grade the search and check the answer; if either fails, search again wider, figures included."""
    best = rerank(question, hybrid(question))
    attempts = [{"search": [question], "best": best[0]["rerank"]}]
    result = answer(question, best) if best[0]["rerank"] >= GOOD_ENOUGH else None
    if result is None or "could not find" in result["answer"].lower():
        searches = rewrites(question)
        wider = fuse(*(hybrid(s) for s in searches))[:12] + pictures(question, k=3)
        best = rerank(question, wider)
        attempts.append({"search": searches, "best": best[0]["rerank"]})
        result = answer(question, best)
    result["attempts"] = attempts
    return result


def record(ref: str) -> dict:
    """Any chunk or figure by its id: n for naive chunks, t for text chunks, p for pictures."""
    return next(r for name in ("naive", "text", "pictures") for r in store(name).records if r["id"] == ref)
