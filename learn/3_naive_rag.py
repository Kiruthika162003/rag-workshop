"""Step 3. Naive RAG: the version in most tutorials, and where it goes wrong.

Chop the text into 500-character pieces, find the three most similar to the question, paste them
into a prompt. No dates, no pictures, no idea how documents relate to each other.

Run: python learn/3_naive_rag.py
     python learn/3_naive_rag.py "What are the winter opening hours?"
"""

import sys

from core.gemini import ask, embed
from core.rag import NAIVE_PROMPT
from core.store import Store

question = " ".join(sys.argv[1:]) or "The dock from incident IR-2026-031 has just been repaired. Who has to sign it off, and what is their phone number?"
print(f"\nQuestion: {question}")

found = Store("naive").search(embed(question), k=3)
print("\nThe three most similar chunks:")
for chunk in found:
    print(f"    {chunk['score']:.3f}  {chunk['doc']}, page {chunk['page']}:  {' '.join(chunk['text'].split())[:90]}...")

context = "\n\n".join(chunk["text"] for chunk in found)
print("\nAnswer:\n   ", ask(NAIVE_PROMPT.format(context=context, question=question)))
print("\nNothing told it which file is newer, what is in the pictures, or that Monument Square changed managers.")
