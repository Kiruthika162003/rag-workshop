"""Contextual chunk headers: a chunk on its own often does not say what it is about.

"We used warmup_steps = 4000" does not say that it is about training, or about the optimizer.
When the index was built, every chunk got a first line naming the paper and the section path it
sits in, like "5 Training / 5.3 Optimizer", and the header is embedded with the chunk. A question
that uses the section's words can now find a chunk that never uses them itself.

The code: text_chunks() and with_sections() in build_index.py

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
