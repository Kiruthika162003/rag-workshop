"""Retrieval metrics: measure the search on its own, before any answer is written.

If the right chunk is not retrieved, no prompt can fix the answer. So retrieval gets its own test:
sixteen questions, each with words that only the chunks answering it contain (FINDS in
core/questions.py). A chunk is relevant if it contains those words. Every way of searching runs on all:

    hit@4         was any relevant chunk in the top 4, the number the answer step sees?
    precision@4   what share of the top 4 is relevant?
    recall@4      what share of all the relevant chunks made the top 4?
    MRR           1 / the place of the first relevant chunk, averaged: 1 is always first
    nDCG@4        like recall, but a relevant chunk at place 1 counts more than one at place 4

Precision@4 stays low here, and that is not a failure: most questions have only one or two relevant
chunks, so even a perfect search cannot fill four places with them. That is why a single number is
never enough.

Cost: the searches run on your computer and cost nothing. Add --rerank for a row that asks Gemini
to rerank once per question: sixteen small calls.

The code: core/metrics.py

Run: python learn/7_evaluation/01_retrieval_metrics.py
     python learn/7_evaluation/01_retrieval_metrics.py --rerank
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from core.metrics import hit_at, ndcg_at, precision_at, recall_at, reciprocal_rank
from core.questions import FINDS
from core.rerank import rerank
from core.search import by_keywords, by_meaning, hybrid
from core.show import step
from core.store import store

METHODS = {"meaning": by_meaning, "keywords": by_keywords, "hybrid": hybrid}
if "--rerank" in sys.argv:
    METHODS["hybrid + rerank"] = lambda q: rerank(q, hybrid(q), keep=12, cutoff=0)
K = 4


def measure(search, question: str, words: str) -> list[float]:
    relevant = [words in chunk["text"] for chunk in search(question)]
    total = sum(words in record["text"] for record in store("text").records)
    return [hit_at(relevant, K), precision_at(relevant, K), recall_at(relevant, K, total),
            reciprocal_rank(relevant), ndcg_at(relevant, K, total)]


step(f"{'method':18}{'hit@4':>8}{'prec@4':>8}{'recall@4':>10}{'MRR':>7}{'nDCG@4':>8}")
for name, search in METHODS.items():
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda test: measure(search, *test), FINDS))
    hit, precision, recall, mrr, ndcg = (sum(column) / len(rows) for column in zip(*rows))
    print(f"{name:18}{hit:8.2f}{precision:8.2f}{recall:10.2f}{mrr:7.2f}{ndcg:8.2f}")
