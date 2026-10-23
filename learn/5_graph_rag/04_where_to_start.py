"""Where to start: find the nodes the question is about.

Two ways in, used together:
    1. names in the question     any node whose whole name appears in the question, like "Table 3"
    2. names in the best chunks  search finds the two closest chunks, and the names in their facts
                                 are the way in when the question's words are not node names

The code: named_in() and from_chunks() in core/graph.py, graph_facts() in core/rag.py

Run: python learn/5_graph_rag/04_where_to_start.py
     python learn/5_graph_rag/04_where_to_start.py "What does the decoder add to each layer?"
"""

from core.graph import graph
from core.search import hybrid
from core.show import body, question, step

asked = question("Which model is listed on the bottom line of Table 3, and what BLEU score does it reach on English-to-German?")

step("1. Nodes named in the question:")
print("   ", sorted(graph().named_in(asked)) or "none")

entry = hybrid(asked, k=2)
step("2. The two best chunks, and the names in their facts:")
for chunk in entry:
    print(f"    {chunk['id']:4} {body(chunk)[:80]}")
print("   ", sorted(graph().from_chunks({c["id"] for c in entry})))
