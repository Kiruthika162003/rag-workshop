"""Walk: follow the links one hop at a time, and watch the answer come within reach.

A walk is only as good as its links. The paper never states "the bottom line of Table 3 is the
Transformer (big)" as one fact: the chunk says "the big transformer model ... the configuration of
this model is listed in the bottom line of Table 3". So a walk that starts only at "the bottom line
of Table 3" never reaches the model. Graph RAG starts from both ways in (04_where_to_start.py):
the names in the question, and the names in the best chunks, which include "transformer big".

Each hop below prints only the new facts that share a word with the question.
Hubs like "we" are never walked through (02_entities.py).

The code: Graph.walk() in core/graph.py, graph_facts() in core/rag.py

Run: python learn/5_graph_rag/05_walk.py
     python learn/5_graph_rag/05_walk.py "What are the parts of each encoder layer?"
"""

from core.graph import graph, key
from core.rag import graph_facts
from core.show import question, step

asked = question("Which model is listed on the bottom line of Table 3, and what BLEU score does it reach on English-to-German?")
start, _ = graph_facts(asked)
step("The walk starts at: " + ", ".join(sorted(start)))

asked_words = {word for word in key(asked).split() if len(word) > 3}
seen = []
for hops in (1, 2, 3):
    facts = graph().walk(start, hops=hops)
    new = [f for f in facts if f not in seen]
    step(f"Hop {hops}: {len(new)} new facts, {len(facts)} in all. Those that share a word with the question:")
    for fact in new:
        if asked_words & set(key(f"{fact['subject']} {fact['object']}").split()):
            print(f"    {fact['subject']}  --{fact['relation']}-->  {fact['object']}"[:140])
    seen = facts
