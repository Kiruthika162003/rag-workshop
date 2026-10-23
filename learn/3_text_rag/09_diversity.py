"""Diversity (MMR): relevant passages that do not repeat each other.

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
