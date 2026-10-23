"""Keyword search: when the exact word matters more than the meaning.

To an embedding, "2048" and "4096" mean nearly the same thing: a big power of two. Papers are full
of exact symbols and numbers: dff, warmup_steps, P100, 28.4. BM25 scores exact words, weighing each
one by how rare it is across all chunks: "model" is in dozens of chunks and counts for little,
"warmup" is in one or two and counts for a lot. Stop words like "what" and "the" are dropped first.

The code: core/keywords.py

Run: python learn/3_text_rag/05_keyword_search.py
     python learn/3_text_rag/05_keyword_search.py "P100"
"""

from core.keywords import words
from core.search import by_keywords, by_meaning
from core.show import chunks, question, step
from core.store import store

asked = question("warmup_steps")

step("The words BM25 searches for, and how rare each one is (rarer counts for more):")
rarity = store("text").keywords.rarity
for word in words(asked):
    print(f"    {rarity.get(word, 0):5.2f}  {word}" + ("   (in no chunk)" if word not in rarity else ""))

for name, search in (("By meaning (vector search)", by_meaning), ("By keywords (BM25)", by_keywords)):
    found = search(asked, k=4)
    step(f"{name}. The winner leads the next one by {found[0]['score'] - found[1]['score']:.3f}:")
    chunks(found)
