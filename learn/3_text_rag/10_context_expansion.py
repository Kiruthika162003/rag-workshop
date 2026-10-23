"""Context expansion: find a small chunk, answer with it and its neighbours on the same page.

Say: Search with small chunks, because they are precise. Answer with more context: each chunk
     plus its neighbours on the same page, so the page number it cites stays right.

Code: with_neighbours() in core/search.py

Run: python learn/3_text_rag/10_context_expansion.py
     python learn/3_text_rag/10_context_expansion.py "How was the big model trained?"
"""

from core.search import hybrid, with_neighbours
from core.show import question, step

asked = question("What learning rate did they use?")
found = hybrid(asked, k=1)

step("What search found:")
print("    | " + found[0]["text"].replace("\n", "\n    | "))
step("What the model reads, with the chunks beside it on the same page:")
print("    | " + with_neighbours(found)[0]["text"].replace("\n", "\n    | "))
