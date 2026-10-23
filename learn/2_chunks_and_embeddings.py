"""Step 2. Chunks and embeddings: the two ideas every RAG is built on.

A chunk is the piece of a document that gets retrieved. An embedding turns a chunk into a list of
numbers, so that pieces with a similar meaning end up close together.

Run: python learn/2_chunks_and_embeddings.py
"""

from core.chunking import fixed_chunks
from core.documents import raw_text
from core.gemini import embed
from core.rag import find_media
from core.store import Store

POLICY = "refund-policy-2026.pdf"

print("\nFixed chunks, every 150 characters, cut wherever they land:")
for chunk in fixed_chunks(raw_text(POLICY)[0][1], size=150)[:3]:
    print("   |", " ".join(chunk.split()), "|")

print("\nSemantic chunks of the same page, from the index. Each one starts with where it came from:")
for chunk in Store("text").records:
    if chunk["doc"] == POLICY:
        print("   |", chunk["text"].replace("\n", " / ")[:110])

question = "How long do I have to report a broken bike?"
print(f"\nHow close is each sentence to the question \"{question}\"?  (1.0 = same meaning)")
vector = embed(question)
for sentence in ["Report a faulty bike in the app within 60 minutes of unlocking it.",
                 "Annual memberships cost $95.",
                 "The board approved four stations in Albany."]:
    print(f"    {float(vector @ embed(sentence)):.3f}  {sentence}")

print("\nThe same space holds pictures, audio and video. Which file matches \"rides per month in 2025\"?")
for record in Store("media").search(embed("rides per month in 2025"), k=3):
    print(f"    {record['score']:.3f}  {record['type']:8} {record['doc']}")

print("\nThe voicemail beat the chart. Scores between a sentence and a recording run high: the modality gap.")
print("Scoring each file as itself and by its description, the better of the two, fixes the order:")
for record in find_media("rides per month in 2025", k=3):
    print(f"    {record['score']:.3f}  {record['type']:8} {record['doc']}")
