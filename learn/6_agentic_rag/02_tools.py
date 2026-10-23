"""Tools: plain functions; the agent reads only their names and docstrings.

Run: python learn/6_agentic_rag/02_tools.py
"""

import inspect

from core.agent import look_at, root_agent, search_figures, search_graph, search_text
from core.show import step

step("What the model is told about each tool:")
for tool in root_agent.tools:
    print(f"\n    {tool.__name__}{inspect.signature(tool)}")
    print("        " + inspect.getdoc(tool).replace("\n", "\n        "))

step("search_text('optimizer', section='5'):")
for hit in search_text("optimizer", section="5")[:2]:
    print(f"    {hit['ref']:4} p{hit['page']}  {hit['text'].splitlines()[-1][:80]}")

step("search_figures('scaled dot-product attention'):")
for hit in search_figures("scaled dot-product attention")[:2]:
    print(f"    {hit['ref']:4} p{hit['page']}  {hit['text'].split('.')[0][:80]}")

step("look_at('p2', 'Which step is marked optional?'):")
print("   ", look_at("p2", "Which step is marked optional?"))

step("search_graph('What does Transformer (big) score?'), the first five facts:")
for fact in search_graph("What does Transformer (big) score?")["facts"][:5]:
    print("   ", fact)
