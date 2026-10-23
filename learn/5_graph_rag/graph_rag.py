"""Graph RAG: a search finds where to start, the knowledge graph connects the rest.

    1. start     names in the question, and in the two best chunks          04_where_to_start.py
    2. walk      three hops along the links, never through a hub             05_walk.py
    3. answer    the facts, and the chunks they came from, numbered          core/answer.py

The code: graph_rag() in core/rag.py

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
