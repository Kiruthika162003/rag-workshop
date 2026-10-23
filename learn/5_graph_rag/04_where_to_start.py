"""Where to start: names in the question, and names in the two best chunks.

Say: Where does the walk begin? At the names in the question, and the names in the two best chunks.
     We need both: the question rarely uses the graph's exact names.

Code: named_in() and from_chunks() in core/graph.py, graph_facts() in core/rag.py

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
