"""Agentic RAG: an agent picks the retrieval, the filters and the order, then checks its work.

The agent has the four tools from 02_tools.py and the instruction AGENT in core/prompts.py. Nothing
tells it which tool to use for which question. It reads the question and decides, sometimes calling
several tools at once, sometimes one after another when the first answer raises a new question.

The code: core/agent.py, built with the Agent Development Kit (ADK)

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
