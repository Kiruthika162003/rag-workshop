"""Ways to find chunks, each one a single function you can try on its own.

    by_meaning       embed the question, compare with every chunk (vector search)
    by_keywords      BM25 on the exact words
    fuse, hybrid     both, merged with reciprocal rank fusion
    diverse          maximal marginal relevance: good matches that do not repeat each other
    with_neighbours  hand the model the chunks around a match, not just the match
    pictures         figures and tables, found by what they look like and by their descriptions
"""

import numpy as np

from core.embeddings import embed_query, embed_query_for_pictures
from core.store import store


def by_meaning(question: str, k: int = 12, index: str = "text", **filters) -> list[dict]:
    return store(index).search(embed_query(question), k, **filters)


def by_keywords(question: str, k: int = 12, index: str = "text", **filters) -> list[dict]:
    return store(index).top(store(index).keywords.scores(question), k, **filters)


def fuse(*rankings: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal rank fusion. Scores from different searches are not comparable, ranks are:
    each list gives a chunk 1 / (60 + its place), and the totals decide the new order."""
    total, chunks = {}, {}
    for ranking in rankings:
        for place, chunk in enumerate(ranking):
            total[chunk["id"]] = total.get(chunk["id"], 0) + 1 / (k + place + 1)
            chunks.setdefault(chunk["id"], chunk)
    return [{**chunks[ref], "score": round(total[ref], 4)} for ref in sorted(total, key=total.get, reverse=True)]


def hybrid(question: str, k: int = 12, **filters) -> list[dict]:
    return fuse(by_meaning(question, k, **filters), by_keywords(question, k, **filters))[:k]


def diverse(question: str, candidates: list[dict], k: int = 4, balance: float = 0.7, index: str = "text") -> list[dict]:
    """Pick, one at a time, the candidate most like the question and least like what is already picked.
    balance 1.0 is plain similarity; lower values trade relevance for variety."""
    vectors = store(index).vectors[[store(index).row[c["id"]] for c in candidates]]
    relevance = vectors @ embed_query(question)
    picked = []
    while len(picked) < min(k, len(candidates)):
        repeats = (vectors @ vectors[picked].T).max(axis=1) if picked else np.zeros(len(candidates))
        value = balance * relevance - (1 - balance) * repeats
        value[picked] = -np.inf
        picked.append(int(np.argmax(value)))
    return [candidates[i] for i in picked]


def with_neighbours(chunks: list[dict], index: str = "text") -> list[dict]:
    """Small to big: search finds the precise chunk, the model reads it with the chunks beside it.
    Only neighbours on the same page join, so the page number it cites stays right."""
    records, row = store(index).records, store(index).row
    widened = []
    for chunk in chunks:
        i = row[chunk["id"]]
        beside = [records[j] for j in (i - 1, i, i + 1) if 0 <= j < len(records)
                  and (records[j]["doc"], records[j]["page"]) == (chunk["doc"], chunk["page"])]
        header = chunk["text"].split("\n", 1)[0]
        widened.append({**chunk, "text": "\n".join([header] + [r["text"].split("\n", 1)[-1] for r in beside])})
    return widened


def pictures(question: str, k: int = 2) -> list[dict]:
    """Two searches, fused by rank. CLIP compares the question with what each figure looks like;
    the text model compares it with the figure's description. CLIP scores sit near 0.3 and text
    scores near 0.7, so the scores cannot be mixed; their ranks can."""
    looks = store("pictures").search(embed_query_for_pictures(question), k=len(store("pictures").records))
    described = store("descriptions").search(embed_query(question), k=len(store("descriptions").records))
    return fuse(looks, described)[:k]
