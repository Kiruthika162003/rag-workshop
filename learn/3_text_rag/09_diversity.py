"""Diversity: four passages that say the same thing waste three places in the prompt.

The paper says that the Transformer beats earlier models in the abstract, the introduction, the
results and the conclusion, in nearly the same words. Maximal marginal relevance picks one passage
at a time: the one most like the question and least like the passages already picked. "repeats" is
how close each pick is to the closest one picked before it. Balance is a dial: turn it too far
towards variety and weaker passages come in.

The code: diverse() in core/search.py

Run: python learn/3_text_rag/09_diversity.py
     python learn/3_text_rag/09_diversity.py "Why use self-attention?"
"""

from core.search import diverse, hybrid
from core.show import body, question, step, where
from core.store import store

asked = question("How well does the Transformer translate English to German?")
candidates = hybrid(asked, k=15)
vector = {c["id"]: store("text").vectors[store("text").row[c["id"]]] for c in candidates}

for balance in (1.0, 0.6):
    step(f"balance {balance}" + ("  (similarity only)" if balance == 1 else "  (relevance and variety)"))
    picked = diverse(asked, candidates, k=4, balance=balance)
    for n, c in enumerate(picked):
        repeats = max((float(vector[c["id"]] @ vector[p["id"]]) for p in picked[:n]), default=0)
        print(f"    repeats {repeats:.2f}  {where(c)}  {body(c)[:52]}")
