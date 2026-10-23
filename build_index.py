"""Read the paper once and build the indexes the lessons and the app search.

    index/naive.*         fixed 500-character chunks of the raw text, the tutorial way
    index/text.*          semantic chunks, each with a header naming the paper and its section
    index/pictures.*      every figure and table, embedded as a picture with CLIP
    index/descriptions.*  the same figures and tables, embedded by their descriptions
    index/graph.json      facts read from all of the above, each pointing back to where it came from

Embeddings run on this computer. Gemini is called once per figure, to describe it, and once per
chunk, to read its facts: about a hundred small calls, a few cents. Every Gemini answer is kept in
index/gemini-cache.json, keyed by exactly what was sent, so building again only pays for what changed.

Run: python build_index.py
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from core.chunking import fixed_chunks, semantic_chunks
from core.documents import library, pieces, raw_text
from core.embeddings import embed_passages, embed_picture
from core.gemini import ask, part
from core.graph import Graph, read_facts
from core.prompts import DESCRIBE, READ_FACTS
from core.store import INDEX, Store

NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\s")
SECTIONS = ("Abstract", "References", "Attention Visualizations")
CACHE = INDEX / "gemini-cache.json"
memory = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def remembered(work, *sent):
    """Gemini's answer to exactly this input: from the cache if it was asked before, else asked now."""
    digest = hashlib.sha256(b"".join(s if isinstance(s, bytes) else s.encode() for s in sent)).hexdigest()
    if digest not in memory:
        memory[digest] = work()
    return memory[digest]


def parallel(work, items):
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(work, items))


def with_sections(found: list[dict]) -> list[dict]:
    """Give every piece the section it is in, like "5 Training / 5.2 Hardware and Schedule"."""
    path = []
    for piece in found:
        if piece["type"] == "heading" and (NUMBERED.match(piece["text"]) or piece["text"] in SECTIONS):
            number = NUMBERED.match(piece["text"])
            depth = number.group(1).count(".") + 1 if number else 1
            path = path[:depth - 1] + [piece["text"]]
        piece["section"] = " / ".join(path) or "Title page"
    return found


def numbered(prefix: str, records: list[dict]) -> list[dict]:
    return [{"id": f"{prefix}{n}", **record} for n, record in enumerate(records)]


def naive_chunks(files):
    return numbered("n", [{**file, "page": page, "section": "", "type": "text", "text": chunk}
                          for file in files for page, text in raw_text(file["doc"]) for chunk in fixed_chunks(text)])


def text_chunks(file, found):
    # The reference list is two pages of author names and titles: noise for every question.
    written = [p for p in found if "text" in p and not p["section"].startswith("References")]
    groups = semantic_chunks(written, embed_passages([p["text"] for p in written]))
    chunks = []
    for group in groups:
        body = " ".join(p["text"] for p in group if p["type"] != "heading")
        if body:
            chunks.append({**file, "page": group[0]["page"], "section": group[-1]["section"],
                           "type": "text", "text": f"{file['title']} · {group[-1]['section']}\n{body}",
                           "find": [f for p in group for f in p["find"]]})
    return chunks


def picture_items(file, found):
    return [{**file, "page": p["page"], "section": p["section"], "type": "picture", "mime": p["mime"], "box": p["box"],
             "data": p["data"]} for p in found if p["type"] == "picture"]


def describe(figure: dict) -> dict:
    """Save the picture next to the index, and ask Gemini once to describe it."""
    data = figure.pop("data")
    (INDEX / figure["media"]).write_bytes(data)
    return {**figure, "text": remembered(lambda: ask([part(data, figure["mime"]), DESCRIBE]), DESCRIBE, data)}


def facts_of(chunk: dict) -> list[dict]:
    found = remembered(lambda: [{k: v for k, v in f.items() if k != "source"} for f in read_facts(chunk)],
                       READ_FACTS.format(**chunk))
    return [{**fact, "source": chunk["id"]} for fact in found]


def main():
    files = library()
    naive = naive_chunks(files)
    Store.save("naive", naive, embed_passages([c["text"] for c in naive]))
    print(f"naive         {len(naive)} fixed-size chunks")

    text, figures = [], []
    for file in files:
        found = with_sections(pieces(file["doc"]))
        text += text_chunks(file, found)
        figures += picture_items(file, found)
    text = numbered("t", text)
    (INDEX / "media").mkdir(parents=True, exist_ok=True)
    figures = parallel(describe, [{**f, "media": f"media/{f['id']}.png"} for f in numbered("p", figures)])
    Store.save("text", text, embed_passages([c["text"] for c in text]))
    print(f"text          {len(text)} semantic chunks")

    Store.save("pictures", figures, [embed_picture((INDEX / f["media"]).read_bytes()) for f in figures])
    Store.save("descriptions", figures, embed_passages([f["text"] for f in figures]))
    print(f"pictures      {len(figures)} figures and tables, embedded as pictures and by their descriptions")

    facts = [fact for found in parallel(facts_of, text + figures) for fact in found]
    Graph.save(facts)
    CACHE.write_text(json.dumps(memory, indent=1), encoding="utf-8")
    print(f"graph         {len(facts)} facts")


if __name__ == "__main__":
    main()
