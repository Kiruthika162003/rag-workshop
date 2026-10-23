"""Graph RAG: find where to start, walk three hops, answer from the facts.

Say: Find where to start, walk three hops, and answer from the facts and the chunks they came from.
     This answers questions no single chunk can: the model that scored 28.4, its size, its training.

Code: graph_rag() in core/rag.py

Run: python learn/5_graph_rag/graph_rag.py
     python learn/5_graph_rag/graph_rag.py "What are the parts of each encoder layer?"
"""

import time

from core.graph import key
from core.rag import graph_rag
from core.show import question, sources, step

asked = question("Which model is listed on the bottom line of Table 3, and what BLEU score does it reach on English-to-German?")
started = time.time()
result = graph_rag(asked)

step("1. Where the walk starts: " + ", ".join(result["graph"]["start"]))

said = key(result["answer"])
step(f"2. The walk found {len(result['graph']['facts'])} facts. The path the answer follows, both ends named in it:")
for fact in result["graph"]["facts"]:
    if key(fact["subject"]) in said and key(fact["object"]) in said:
        print(f"    [{fact['n']}] {fact['subject']}  --{fact['relation']}-->  {fact['object']}")

sources(result)
print(f"\n{time.time() - started:.1f} seconds")
