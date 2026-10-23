"""Retrieval metrics: hit@4, precision@4, recall@4, MRR and nDCG for each search. Free, on the laptop.

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
