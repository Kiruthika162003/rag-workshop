"""Vector search by hand: score every chunk with one matrix multiplication, keep the best.

Run: python learn/1_foundations/03_vector_search.py
     python learn/1_foundations/03_vector_search.py "Why scale the dot products?"
"""

import numpy as np

from core.embeddings import embed_query
from core.show import body, question, step, where
from core.store import store

index = store("text")
step(f"The index: a {index.vectors.shape[0]} x {index.vectors.shape[1]} matrix, one row per chunk, and {len(index.records)} records.")
print("    One record:", {k: v for k, v in index.records[0].items() if k in ("id", "doc", "page", "section", "type")})

asked = question("What optimizer was used for training?")
scores = index.vectors @ embed_query(asked)       # one similarity per chunk
best = np.argsort(-scores)[:5]                     # the five highest
step("scores = vectors @ question;  best = argsort(-scores)[:5]")
for i in best:
    print(f"    {scores[i]:.3f}  {index.records[i]['id']:4} {where(index.records[i])}  {body(index.records[i])[:56]}")
