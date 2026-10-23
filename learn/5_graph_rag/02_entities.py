"""Entities: one spelling per name, names inside names, and hubs that are never walked through.

Say: The model writes "Transformer (big)" one time and "transformer big" the next. We give each
     name one spelling, link names inside names, and skip hubs like "we" that touch everything.

Code: key(), is_proper(), Graph.part_of and Graph.hubs in core/graph.py

Run: python learn/5_graph_rag/02_entities.py
"""

from core.graph import HUB, graph, key
from core.show import step

step("1. One spelling:")
for name in ("Transformer (big)", "transformer big", "the Transformer (big) model."):
    print(f"    {name!r:32} -> {key(name)!r}")

step("2. Nodes that contain a proper name, and are linked to it (the first fifteen):")
for node, names in [(node, names) for node, names in graph().part_of.items() if names][:15]:
    print(f"    {node[:60]:60} -> {', '.join(sorted(names))}")

step(f"3. Hubs, with more than {HUB} facts each:")
for node in sorted(graph().hubs, key=lambda n: -len(graph().links[n])):
    print(f"    {len(graph().links[node]):>3}  {node}")
