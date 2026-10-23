"""Keyword search (BM25): exact words win where embeddings blur, like warmup_steps or 2048.

Say: Embeddings blur exact words: to a vector, 2048 and 4096 look alike. Papers are full of symbols.
     BM25 scores exact words and counts rare ones more. For warmup_steps, keywords win by 7.0,
     vectors by only 0.07.

Code: core/keywords.py

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
