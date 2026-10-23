"""Step 7. Agentic RAG: an agent picks the retrieval, the filters and the order, then checks its work.

The agent (core/agent.py) has four tools: search_documents, search_media, look_at and search_graph.
Nothing tells it which to use. It reads the question and decides, sometimes calling several at once.

Run: python learn/7_agentic_rag.py
     python learn/7_agentic_rag.py "What are the winter opening hours, and what happens in a snow emergency?"
"""

import sys

from core.agent import ask_agent

question = " ".join(sys.argv[1:]) or "The dock from incident IR-2026-031 has just been repaired. Who has to sign it off, and what is their phone number?"
print(f"\nQuestion: {question}")

result = ask_agent(question)
print("\nWhat the agent decided to do:")
for step in result["steps"]:
    print(f"    {step['tool']}({', '.join(f'{k}={v!r}' for k, v in step['args'].items())})")

print("\nAnswer:\n   ", result["answer"])
print("\nSources it cited:")
for source in result["sources"]:
    print(f"    [{source['id']}] {source['type']:8} {source['title']}, page {source['page']}")
