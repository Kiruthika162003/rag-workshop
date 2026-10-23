"""The last step of every RAG: turn what was retrieved into an answer that cites it."""

from datetime import date

from google.genai import errors

from core.gemini import ask, part
from core.store import INDEX

RULES = """Today is {today}. Answer the question using only the numbered sources below.
- Put the source number after every fact you use, like [2]. Pictures, recordings and videos are sources too.
- When you use a recording or a video, give the time, like [3, 0:12].
- Sources carry dates. If two sources disagree, trust the most recent one and say the older one is out of date.
- If the sources do not contain the answer, say you could not find it. Never guess.
- Be brief: two to four sentences."""


def prompt(question: str, sources: list[dict], facts, originals: bool) -> list:
    parts = [RULES.format(today=date.today().isoformat())]
    for n, source in enumerate(sources, start=1):
        label = f"[{n}] {source['title']}, page {source['page']}, dated {source['date']}"
        if source["type"] == "text":
            parts.append(f"{label}:\n{source['text']}")
        elif originals:
            parts += [f"{label}. A {source['type']} from this file:", part((INDEX / source["media"]).read_bytes(), source["mime"])]
        else:
            parts.append(f"{label}. The {source['type']}, as described when the index was built:\n{source['text']}")
    if facts:
        parts.append("Facts from the knowledge graph, with the source each was read from:\n" + "\n".join(
            f"- {f['subject']} | {f['relation']} | {f['object']}{' | since ' + f['since'] if f['since'] else ''} | [{f['n']}]"
            for f in facts))
    parts.append(f"Question: {question}")
    return parts


def answer(question: str, sources: list[dict], facts=()) -> dict:
    try:
        reply = ask(prompt(question, sources, facts, originals=True))
    except errors.APIError:
        if all(source["type"] == "text" for source in sources):
            raise
        # Sending a recording or a video is the slowest request there is. If it fails,
        # answer from its transcript or description instead: still grounded, still citable.
        reply = ask(prompt(question, sources, facts, originals=False))
    return {"answer": reply, "sources": [shown(s) for s in sources]}


def shown(source: dict) -> dict:
    """What the terminal and the page need to show a source."""
    keep = ("id", "doc", "title", "kind", "date", "page", "type", "text", "score", "rerank", "find", "box", "media")
    return {k: source[k] for k in keep if k in source}
