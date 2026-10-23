"""Multimodal RAG: text RAG, plus the figures and tables, found by meaning and shown to Gemini.

    1. text      hybrid search and reranking from 3_text_rag, the best three kept
    2. figures   the two best figures or tables, by look and by description    03_two_scales.py
    3. answer    passages as text, figures as pictures, all numbered            05_send_the_original.py

The code: multimodal() in core/rag.py

Run: python learn/4_multimodal_rag/multimodal_rag.py
     python learn/4_multimodal_rag/multimodal_rag.py "According to Table 3, how many parameters does the big model have?"
     python learn/4_multimodal_rag/multimodal_rag.py "In the attention plot for the word 'making', which two words get the strongest attention?"
"""

import time
from pathlib import Path

from core.answer import answer
from core.documents import page_picture
from core.rerank import rerank
from core.search import hybrid, pictures, with_neighbours
from core.show import body, question, sources, step, where

asked = question("In the scaled dot-product attention diagram, which step is optional, and what comes right after it?")
started = time.time()

passages = rerank(asked, hybrid(asked), keep=3)
step("1. Text passages, reranked:")
for c in passages:
    print(f"    {c['rerank']:>2}/10  {where(c)}  {body(c)[:52]}")

found = pictures(asked)
step("2. Figures and tables, by look and by description:")
for figure in found:
    print(f"    {figure['id']}  page {figure['page']:<3} {' '.join(figure['text'].split())[:72]}...")

result = answer(asked, with_neighbours(passages) + found)
sources(result)

Path("out").mkdir(exist_ok=True)
Path("out/figure.png").write_bytes(page_picture(found[0]["doc"], found[0]["page"], box=found[0]["box"]))
print("\nThe page the best figure is on, with the figure outlined: out/figure.png")
print(f"\n{time.time() - started:.1f} seconds")
