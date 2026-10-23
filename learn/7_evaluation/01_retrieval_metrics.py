"""Retrieval metrics: measure the search on its own, before any answer is written.

If the right chunk is not retrieved, no prompt can fix the answer. So retrieval gets its own test:
sixteen questions, each with words that only the chunk answering it contains (FINDS in
core/questions.py). Every way of searching from 3_text_rag runs on all of them:

    hit@1   the right chunk came first
    hit@4   it was in the top four, the number the answer step sees
    MRR     mean reciprocal rank: 1 for first place, 1/2 for second, 1/3 for third, averaged

Read the table; it is the point of this lesson. Some questions use plain words, some the paper's own
symbols, so each search wins some and loses some. Only a measurement tells you which mix to use,
and no technique is "better" until it is measured on your own questions.

Cost: the three searches run on your computer and cost nothing. Add --rerank for a fourth row that
asks Gemini to rerank once per question: sixteen small calls.

The code: core/search.py, core/rerank.py

Run: python learn/7_evaluation/01_retrieval_metrics.py
     python learn/7_evaluation/01_retrieval_metrics.py --rerank
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from core.questions import FINDS
from core.rerank import rerank
from core.search import by_keywords, by_meaning, hybrid
from core.show import step

METHODS = {
    "meaning": by_meaning,
    "keywords": by_keywords,
    "hybrid": hybrid,
}
if "--rerank" in sys.argv:
    METHODS["hybrid + rerank"] = lambda q: rerank(q, hybrid(q), keep=12, cutoff=0)


def place(found: list[dict], words: str) -> int:
    """1 for first place; 0 when the right chunk was not found at all."""
    return next((n for n, chunk in enumerate(found, start=1) if words in chunk["text"]), 0)


step(f"{'method':18}{'hit@1':>7}{'hit@4':>7}{'MRR':>7}   misses (not in the top four)")
for name, search in METHODS.items():
    with ThreadPoolExecutor(8) as pool:
        places = list(pool.map(lambda test: place(search(test[0]), test[1]), FINDS))
    hit1 = sum(p == 1 for p in places) / len(places)
    hit4 = sum(0 < p <= 4 for p in places) / len(places)
    mrr = sum(1 / p for p in places if p) / len(places)
    misses = [question[:28] for (question, _), p in zip(FINDS, places) if not 0 < p <= 4]
    print(f"{name:18}{hit1:7.2f}{hit4:7.2f}{mrr:7.2f}   {'; '.join(misses)}")
