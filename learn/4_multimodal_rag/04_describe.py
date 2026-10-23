"""Describe: give every figure and table a text twin, once.

When the index is built, Gemini looks at each figure once and writes down every label, number and
connection in it. The description is searched by meaning and by keywords, it is what corrective RAG
and the agent read first, and it is the fallback answer if sending the picture fails. Every answer is
kept in index/gemini-cache.json, so building the index again does not pay for the same picture twice.

The code: describe() in build_index.py, the prompt DESCRIBE in core/prompts.py

Run: python learn/4_multimodal_rag/04_describe.py
     python learn/4_multimodal_rag/04_describe.py p2      one figure in full
"""

import sys

from core.prompts import DESCRIBE
from core.show import step
from core.store import store

step("The prompt:")
print("   ", DESCRIBE)

chosen = sys.argv[1] if len(sys.argv) > 1 else None
for record in store("pictures").records:
    if chosen in (None, record["id"]):
        step(f"{record['id']}  page {record['page']}, {record['section']}:")
        text = record["text"] if chosen else record["text"][:300] + " ..."
        print("    | " + text.replace("\n\n", "\n").replace("\n", "\n    | "))
