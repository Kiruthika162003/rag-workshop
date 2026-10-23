"""Read facts: turn each chunk into (subject, relation, object) links.

When the index is built, Gemini reads every chunk and every figure description once and writes down
the facts it states, each one pointing back to the chunk it came from. This lesson reads one chunk
again, live, so you can see the prompt and what comes back.

The code: read_facts() in core/graph.py, the prompt READ_FACTS in core/prompts.py

Run: python learn/5_graph_rag/01_read_facts.py
     python learn/5_graph_rag/01_read_facts.py t48      any chunk id, from 1_foundations/03_vector_search.py
"""

import sys

from core import prompts
from core.graph import graph, read_facts
from core.rag import record
from core.show import step
from core.store import store

bottom_line = next(r["id"] for r in store("text").records if "bottom line of Table 3" in r["text"])
chunk = record(sys.argv[1] if len(sys.argv) > 1 else bottom_line)

step("The prompt, filled in for one chunk:")
print("    | " + prompts.READ_FACTS.format(**chunk).replace("\n", "\n    | "))

step("What Gemini wrote down:")
for fact in read_facts(chunk):
    print(f"    {fact['subject']}  --{fact['relation']}-->  {fact['object']}")

step(f"Every chunk was read this way when the index was built: {len(graph().facts)} facts in index/graph.json.")
