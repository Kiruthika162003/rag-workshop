"""Hybrid search: run vector search and keyword search, then merge the two lists.

Their scores cannot be added: a cosine of 0.7 and a BM25 of 7.2 are not on one scale. Reciprocal
rank fusion uses only the places: each list gives a chunk 1 / (60 + its place), and the totals
decide. A chunk near the top of both lists beats a chunk at the top of only one.

The code: fuse() and hybrid() in core/search.py

Run: python learn/3_text_rag/06_hybrid_search.py
     python learn/3_text_rag/06_hybrid_search.py "beam size and length penalty"
"""

from core.search import by_keywords, by_meaning, fuse
from core.show import body, question, step, where

asked = question("What learning rate schedule with warmup_steps did they use?")
meaning, keywords = by_meaning(asked, k=6), by_keywords(asked, k=6)
place = {name: {c["id"]: n for n, c in enumerate(found, start=1)} for name, found in (("meaning", meaning), ("keywords", keywords))}

step(f"    {'fused':>7}  {'meaning':>7}  {'keywords':>8}  chunk")
for chunk in fuse(meaning, keywords)[:6]:
    ranks = [str(place[name].get(chunk["id"], "-")) for name in ("meaning", "keywords")]
    print(f"    {chunk['score']:.4f}  {ranks[0]:>7}  {ranks[1]:>8}  {where(chunk)}  {body(chunk)[:44]}")
