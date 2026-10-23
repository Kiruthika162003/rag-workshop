"""Explore the graph: every fact about one thing, from every section.

Say: Search sees one chunk at a time. The graph sees that Transformer (big) appears in three sections,
     and that those facts are about the same model.

Code: Graph.links in core/graph.py

Run: python learn/5_graph_rag/03_explore_the_graph.py
     python learn/5_graph_rag/03_explore_the_graph.py "multi-head attention"
"""

import sys

from core.graph import graph, key
from core.rag import record
from core.show import step

node = key(" ".join(sys.argv[1:]) or "Transformer (big)")
facts = graph().links[node] + [f for other, names in graph().part_of.items() if node in names for f in graph().links[other]]
step(f"{len(facts)} facts touch '{node}':")
for fact in facts:
    source = record(fact["source"])
    print(f"    {fact['subject']}  --{fact['relation']}-->  {fact['object']}   [p{source['page']}, {source['section'].split(' / ')[-1]}]")
