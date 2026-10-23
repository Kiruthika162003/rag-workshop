"""Entities: decide when two names are the same thing, and which names are useless.

The model writes "Transformer (big)" in one fact, "transformer big" in another and "the bottom line of
Table 3" in a third. Unless they meet in one node, the graph has islands and no path between them.

    1. one spelling   lower case, no punctuation:  key()
    2. names inside   a node whose name contains a proper name, like "the bottom line of Table 3",
                      is linked to that name too:  Graph.part_of
    3. hubs           "we" and "transformer" touch dozens of facts. A walk through them reaches
                      everything, so it never passes through one:  Graph.hubs

The code: key(), is_proper(), Graph.part_of and Graph.hubs in core/graph.py

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
