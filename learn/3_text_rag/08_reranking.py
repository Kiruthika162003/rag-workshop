"""Reranking: read the candidates before trusting them.

Search is fast and rough: it finds twelve chunks that look like the question. A reranker is slow
and careful: it reads each candidate next to the question and scores whether it answers it. The
best four are kept, and any that score under 3 out of 10 are dropped, so the answer step never
sees passages that only sound relevant.

The code: core/rerank.py, the prompt RERANK in core/prompts.py

Run: python learn/3_text_rag/08_reranking.py
     python learn/3_text_rag/08_reranking.py "What beam size was used?"
"""

from core import prompts
from core.rerank import rerank
from core.search import hybrid
from core.show import body, question, step, where

asked = question("Why are the dot products scaled by the square root of dk?")
candidates = hybrid(asked)

step("1. The prompt (RERANK in core/prompts.py), with the first two of the twelve candidates:")
first_two = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(candidates[:2]))
print("    | " + prompts.RERANK.format(question=asked, passages=first_two + "\n\n...").replace("\n", "\n    | "))

scored = rerank(asked, candidates, keep=len(candidates), cutoff=0)
order = {c["id"]: n for n, c in enumerate(candidates, start=1)}
step("2. Every candidate: its place from search, and the reranker's score out of 10:")
for c in scored:
    print(f"    search #{order[c['id']]:<3} {c['rerank']:>2}/10  {where(c)}  {body(c)[:46]}")

step("3. Kept for the answer: the best four that score 3 or more.")
for c in [c for c in scored[:4] if c["rerank"] >= 3] or scored[:1]:
    print(f"    {c['rerank']:>2}/10  {body(c)[:90]}")
