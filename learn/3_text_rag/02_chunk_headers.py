"""Chunk headers: each chunk starts with its section, so the section's words can find it.

Say: "We used warmup_steps = 4000" never says it is about the optimizer. So every chunk starts
     with its section. Same search, with and without headers: the right chunk moves up.

Code: text_chunks() and with_sections() in build_index.py

Run: python learn/3_text_rag/02_chunk_headers.py
     python learn/3_text_rag/02_chunk_headers.py "What regularization did they use?"
"""

import numpy as np

from core.embeddings import embed_passages, embed_query
from core.show import body, question, step, where
from core.store import store

asked = question("How was the optimizer set up?")
records = store("text").records
bodies = embed_passages([body(r) for r in records])

for name, vectors in (("Chunks without headers:", bodies), ("Chunks with headers, as they are stored:", store("text").vectors)):
    scores = vectors @ embed_query(asked)
    step(name)
    for i in np.argsort(-scores)[:3]:
        print(f"    {scores[i]:.3f}  {where(records[i])}  {body(records[i])[:56]}")

step("One header:")
print("   ", records[int(np.argmax(store("text").vectors @ embed_query(asked)))]["text"].split("\n")[0])
