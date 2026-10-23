"""Hybrid search: vector and keyword results, merged by rank.

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
