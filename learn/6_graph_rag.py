"""Step 6. Graph RAG: connect facts that live in different files.

When the index was built, Gemini read every chunk and wrote down its facts as links:
(subject, relation, object), each pointing back to the chunk it came from. Answering walks those
links: from the incident, to the station, to whoever manages it now, to their phone number.

Run: python learn/6_graph_rag.py
     python learn/6_graph_rag.py --node "Monument Square"     every link of one thing
"""

import argparse

from core.graph import key
from core.rag import graph, graph_rag


def link(fact: dict) -> str:
    since = f"   (since {fact['since']})" if fact["since"] else ""
    return f"{fact['subject']}  --{fact['relation']}-->  {fact['object']}{since}"


options = argparse.ArgumentParser()
options.add_argument("question", nargs="*", default=[
    "The dock from incident IR-2026-031 has just been repaired. Who has to sign it off, and what is their phone number?"])
options.add_argument("--node", help="show every fact linked to this name instead of asking")
args = options.parse_args()

if args.node:
    for fact in graph().links[key(args.node)]:
        print("   ", link(fact))
    raise SystemExit

question = " ".join(args.question)
print(f"\nQuestion: {question}  ({len(graph().facts)} facts in the graph)")

result = graph_rag(question)
print("\nWhere the walk starts:", ", ".join(result["graph"]["start"]))

said = key(result["answer"])
print(f"\nThe walk found {len(result['graph']['facts'])} facts. The path the answer follows, both ends named in it:")
for fact in result["graph"]["facts"]:
    if key(fact["subject"]) in said and key(fact["object"]) in said:
        print(f"    [{fact['n']}] {link(fact)}")

print("\nAnswer:\n   ", result["answer"])
print("\nSources:")
for n, source in enumerate(result["sources"], start=1):
    print(f"    [{n}] {source['title']}, page {source['page']}")
