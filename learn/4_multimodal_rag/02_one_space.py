"""One space for pictures and words: CLIP.

CLIP was trained on hundreds of millions of pictures with their captions, so that a picture and a
sentence describing it end up close together. It runs on your computer. A typed question can now
find a figure by what it looks like, and a figure can be the query: here Figure 1 itself is
compared with a few descriptions, and with the other figures.

The code: embed_picture() and embed_query_for_pictures() in core/embeddings.py

Run: python learn/4_multimodal_rag/02_one_space.py
     python learn/4_multimodal_rag/02_one_space.py "lines connecting words in two rows"
"""

from core.embeddings import PICTURE_MODEL, embed_query_for_pictures
from core.show import question, step
from core.store import store

figures = store("pictures")
asked = question("a diagram of boxes and arrows")
step(f"Words in, figures out ({PICTURE_MODEL}):")
for figure in figures.search(embed_query_for_pictures(asked), k=len(figures.records)):
    print(f"    {figure['score']:.3f}  {figure['id']}  page {figure['page']:<3} {figure['section'].split(' / ')[-1]}")

figure_1 = figures.vectors[0]
step("A picture in: Figure 1 compared with words, and with the other figures.")
for words in ("the architecture of a neural network", "a table of numbers", "a photo of a cat"):
    print(f"    {float(embed_query_for_pictures(words) @ figure_1):.3f}  \"{words}\"")
for figure, vector in list(zip(figures.records, figures.vectors))[1:4]:
    print(f"    {float(vector @ figure_1):.3f}  {figure['id']}, page {figure['page']}")
