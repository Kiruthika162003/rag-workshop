"""Walk: follow the links hop by hop until the answer is in reach.

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
