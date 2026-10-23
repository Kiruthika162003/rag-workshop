"""Chunking: the piece of a document that gets retrieved, and why the cut matters.

Fixed chunks cut every N characters, wherever that lands, so sentences break in half and the
end of one topic is glued to the start of the next. Semantic chunks keep neighbouring sentences
together while they talk about the same thing. Two sentences "talk about the same thing" when
their embeddings are close; a new chunk starts at a heading, a page break, a table row, or where
the similarity drops into the lowest quarter for this document.

The code: core/chunking.py

Run: python learn/3_text_rag/01_chunking.py          page 7
     python learn/3_text_rag/01_chunking.py 3        any page
"""

import sys

import numpy as np

from core.chunking import fixed_chunks, semantic_chunks
from core.documents import library, pieces, raw_text
from core.embeddings import embed_passages
from core.show import step

doc, page = library()[0]["doc"], int(sys.argv[1]) if len(sys.argv) > 1 else 7

step(f"Fixed chunks of page {page}, every 200 characters. The | marks a cut:")
for chunk in fixed_chunks(raw_text(doc)[page - 1][1], size=200)[:4]:
    print("    |" + " ".join(chunk.split()) + "|")

written = [p for p in pieces(doc) if "text" in p]
vectors = embed_passages([p["text"] for p in written])
gaps = [float(a @ b) for a, b in zip(vectors, vectors[1:])]

step(f"Semantic chunks of page {page}. The number is how close each piece is to the one before it; "
     f"below {np.percentile(gaps, 25):.3f} counts as a change of topic.")
first = 0
for chunk in semantic_chunks(written, vectors):
    if chunk[0]["page"] == page:
        print("    " + "-" * 92)
        for n, piece in enumerate(chunk, start=first):
            print(f"    {gaps[n - 1] if n else 0:.3f}  {piece['type']:8}  {piece['text'][:72]}")
    first += len(chunk)
