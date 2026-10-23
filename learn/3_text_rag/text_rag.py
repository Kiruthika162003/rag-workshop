"""Text RAG done right: every step from this folder, in one pipeline.

    1. hybrid search      meaning and exact words, fused                       06_hybrid_search.py
    2. metadata filters   section and pages, before ranking                    03, 04
    3. rerank             read the twelve candidates, keep the best four        08_reranking.py
    4. expand             each chunk with its neighbours on the same page       10_context_expansion.py
    5. answer             numbered sources, page numbers, no guessing           11_grounded_answer.py

The code: text() in core/rag.py is these five lines.

Run: python learn/3_text_rag/text_rag.py
     python learn/3_text_rag/text_rag.py "What dropout rate is used?" --section 5
     python learn/3_text_rag/text_rag.py "What does Table 3 show?" --pages 9 9
"""

import argparse
import time

from core.answer import answer
from core.rerank import rerank
from core.search import hybrid, with_neighbours
from core.show import body, chunks, sources, step, where

options = argparse.ArgumentParser()
options.add_argument("question", nargs="*", default=["On what hardware was the big model trained, and for how long?"])
options.add_argument("--section", help="only this section, by number or name, like 5.2 or Training")
options.add_argument("--pages", nargs=2, type=int, help="first and last page")
args = options.parse_args()
asked = " ".join(args.question)
print(f"\nQuestion: {asked}")
started = time.time()

candidates = hybrid(asked, section=args.section, pages=tuple(args.pages) if args.pages else None)
step(f"1-2. Hybrid search, filtered: {len(candidates)} candidates. The top five:")
chunks(candidates[:5])

best = rerank(asked, candidates)
step("3. Reranked, kept:")
for c in best:
    print(f"    {c['rerank']:>2}/10  {where(c)}  {body(c)[:52]}")

step("4-5. Expanded with neighbours, and answered:")
sources(answer(asked, with_neighbours(best)))
print(f"\n{time.time() - started:.1f} seconds")
