"""Step 4. Text RAG done right: semantic chunks, metadata filters, reranking and citations.

    1. retrieve   12 candidates by similarity
    2. filter     by metadata: kind of document, date, newest version only
    3. rerank     Gemini reads each candidate and scores whether it answers the question
    4. answer     with a numbered source after every fact, and the page it came from

Run: python learn/4_text_rag.py
     python learn/4_text_rag.py "What is the charge for a lost bike?" --all-versions
     python learn/4_text_rag.py "What did the board decide?" --kind Minutes --after 2026-04-01
"""

import argparse
from pathlib import Path

from core.answer import answer
from core.documents import DOCS, page_picture
from core.rag import find_text, rerank

options = argparse.ArgumentParser()
options.add_argument("question", nargs="*", default=["How long do I have to report a broken bike to get a refund?"])
options.add_argument("--kind", help="only this kind of document, like Policy or Minutes")
options.add_argument("--after", help="only documents dated on or after YYYY-MM-DD")
options.add_argument("--all-versions", action="store_true", help="keep old versions of replaced documents")
args = options.parse_args()
question = " ".join(args.question)
print(f"\nQuestion: {question}")

candidates = find_text(question, k=12, kinds=[args.kind] if args.kind else None, after=args.after,
                       latest_only=not args.all_versions)
print(f"\n1-2. Retrieved and filtered: {len(candidates)} candidates" + ("" if args.all_versions else ", old versions removed"))
for c in candidates[:5]:
    print(f"    {c['score']:.3f}  {c['date']}  {c['title']}, page {c['page']}")

best = rerank(question, candidates)
print("\n3. Reranked: up to four kept, any that score under 3 dropped:")
for c in best:
    print(f"    {c['rerank']:>2}/10  {c['title']}:  {c['text'].split(chr(10))[-1][:80]}...")

result = answer(question, best)
print("\n4. Answer:\n   ", result["answer"])

print("\nSources:")
Path("out").mkdir(exist_ok=True)
for n, source in enumerate(result["sources"], start=1):
    print(f"    [{n}] {source['title']}, page {source['page']}   {(DOCS / source['doc']).as_uri()}#page={source['page']}")
top = result["sources"][0]
Path("out/source-1.png").write_bytes(page_picture(top["doc"], top["page"], top["find"]))
print("\nSource [1], with the passage highlighted: out/source-1.png")
