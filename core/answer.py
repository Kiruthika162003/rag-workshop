"""The last step of every RAG: turn what was retrieved into an answer that cites it."""

from google.genai import errors

from core import prompts
from core.gemini import ask, part
from core.store import INDEX


def prompt(question: str, sources: list[dict], facts=(), originals: bool = True) -> list:
    """The exact message the model gets: the rules, each source with its number, page and section,
    any graph facts, then the question. Figures and tables go in as pictures."""
    parts = [prompts.ANSWER]
    for n, source in enumerate(sources, start=1):
        label = f"[{n}] page {source['page']}, section {source['section']}"
        if source["type"] == "text":
            parts.append(f"{label}:\n{source['text']}")
        elif originals:
            parts += [f"{label}. A picture of this figure or table:", part((INDEX / source["media"]).read_bytes(), source["mime"])]
        else:
            parts.append(f"{label}. The figure or table, as described when the index was built:\n{source['text']}")
    if facts:
        parts.append("Facts from the knowledge graph, with the source each was read from:\n" + "\n".join(
            f"- {f['subject']} | {f['relation']} | {f['object']} | [{f['n']}]" for f in facts))
    parts.append(f"Question: {question}")
    return parts


def answer(question: str, sources: list[dict], facts=()) -> dict:
    try:
        reply = ask(prompt(question, sources, facts))
    except errors.APIError:
        if all(source["type"] == "text" for source in sources):
            raise
        # A request with pictures is the biggest one there is. If it fails, answer from the
        # descriptions written when the index was built instead: still grounded, still citable.
        reply = ask(prompt(question, sources, facts, originals=False))
    return {"answer": reply, "sources": [shown(s) for s in sources]}


def shown(source: dict) -> dict:
    """What the terminal and the page need to show a source."""
    keep = ("id", "doc", "title", "section", "page", "type", "text", "score", "rerank", "find", "box", "media", "mime")
    return {k: source[k] for k in keep if k in source}
