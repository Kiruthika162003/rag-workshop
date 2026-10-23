"""Context expansion, or small to big: search with small chunks, answer with bigger ones.

Small chunks are easy to find, because each one is about one thing. But the sentence that answers
the question often needs the sentence before it. So the answer step gets each chunk together with
its neighbours on the same page; the page number it cites stays right.

The code: with_neighbours() in core/search.py

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
