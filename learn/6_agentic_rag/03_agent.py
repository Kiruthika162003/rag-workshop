"""Agentic RAG: an agent picks its own tools, in its own order.

Run: python learn/6_agentic_rag/03_agent.py
     python learn/6_agentic_rag/03_agent.py "In the attention plot for the word 'making', which two words get the strongest attention?"
"""

import time

from core.agent import ask_agent
from core.show import question, step

asked = question("In the scaled dot-product attention diagram, which step is optional, and what comes right after it?")
started = time.time()
result = ask_agent(asked)

step("What the agent decided to do:")
for n, call in enumerate(result["steps"], start=1):
    print(f"    {n}. {call['tool']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")

step("Answer:")
print("   ", result["answer"])
step("Sources it cited:")
for source in result["sources"]:
    print(f"    [{source['id']}] {source['type']:8} {source['title']}, page {source['page']}")
print(f"\n{time.time() - started:.1f} seconds")
