"""Read every file once and build the indexes the lessons and the app search.

    index/naive.*   fixed 500-character chunks of the raw text, the tutorial way
    index/text.*    semantic chunks, each with a header naming its document, date and section
    index/media.*   every picture, recording and video, embedded as itself, plus a description or transcript
    index/media-descriptions.*   the same items, embedded by their descriptions
    index/graph.json  facts read from all of the above, each pointing back to where it came from

Run: python build_index.py   (about five minutes, a few cents of Gemini calls)
"""

from concurrent.futures import ThreadPoolExecutor

from core.chunking import fixed_chunks, semantic_chunks
from core.documents import library, pieces, raw_text
from core.gemini import ask, embed, part
from core.graph import Graph, read_facts
from core.store import INDEX, Store

DESCRIBE = {
    "picture": "Describe this picture in plain text, no markdown, so it can be found by search. "
               "Say what kind of picture it is in one sentence, then copy out every number, name, date and label exactly.",
    "audio": "Transcribe this recording word for word. Start each sentence with its time, like [0:07].",
    "video": "Describe what this video shows, in order, in plain text with no markdown. "
             "Start each line with its time, like [0:12], and copy any text on screen exactly.",
}


def parallel(work, items):
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(work, items))


def naive_chunks(files):
    chunks = []
    for file in files:
        for page, text in raw_text(file["doc"]):
            for chunk in fixed_chunks(text):
                chunks.append({"id": f"n{len(chunks)}", **file, "page": page, "type": "text", "text": chunk})
    return chunks


def text_chunks(files):
    chunks = []
    for file in files:
        written = [p for p in pieces(file["doc"]) if "text" in p]
        if not written:
            continue
        groups = semantic_chunks(written, parallel(embed, [p["text"] for p in written]))
        section = ""
        for group in groups:
            section = next((p["text"] for p in group if p["type"] == "heading"), section)
            body = " ".join(p["text"] for p in group if p["type"] != "heading")
            if not body:
                continue
            chunks.append({"id": f"t{len(chunks)}", **file, "page": group[0]["page"], "type": "text",
                           "text": f"{file['title']} ({file['date']}) · {section}\n{body}",
                           "find": [f for p in group for f in p["find"]]})
    return chunks


def describe(item: dict) -> str:
    return ask([part(item["data"], item["mime"]), DESCRIBE[item["type"]]])


def media_items(files):
    (INDEX / "media").mkdir(parents=True, exist_ok=True)
    items = []
    for file in files:
        for piece in pieces(file["doc"]):
            if "data" not in piece:
                continue
            item = {"id": f"m{len(items)}", **file, "page": piece["page"], "type": piece["type"], "mime": piece["mime"]}
            if piece["type"] == "picture":
                item["media"] = f"media/{item['id']}.png"
                item["box"] = piece["box"]
                (INDEX / item["media"]).write_bytes(piece["data"])
            else:
                item["media"] = f"../data/docs/{file['doc']}"
            item["data"] = piece["data"]
            items.append(item)
    for item, text in zip(items, parallel(describe, items)):
        item["text"] = text
    return items


def main():
    files = library()
    print(f"{len(files)} files")

    naive = naive_chunks(files)
    Store.save("naive", naive, parallel(embed, [c["text"] for c in naive]))
    print(f"naive   {len(naive)} fixed-size chunks")

    text = text_chunks(files)
    Store.save("text", text, parallel(embed, [c["text"] for c in text]))
    print(f"text    {len(text)} semantic chunks")

    media = media_items(files)
    records = [{k: v for k, v in item.items() if k != "data"} for item in media]
    Store.save("media", records, parallel(lambda item: embed(item["data"], item["mime"]), media))
    Store.save("media-descriptions", records, parallel(embed, [item["text"] for item in media]))
    print(f"media   {len(media)} pictures, recordings and videos, each embedded as itself and by its description")

    facts = [fact for found in parallel(read_facts, text + media) for fact in found]
    Graph.save(facts)
    print(f"graph   {len(facts)} facts")


if __name__ == "__main__":
    main()
