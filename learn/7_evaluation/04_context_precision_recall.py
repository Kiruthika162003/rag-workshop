"""Context precision and recall: useful sources retrieved first, and holding the whole answer?

Run: python learn/7_evaluation/04_context_precision_recall.py
"""

from core.metrics import context_precision, context_recall
from core.questions import QUESTIONS
from core.rerank import rerank
from core.search import hybrid, pictures
from core.show import step

test = next(q for q in QUESTIONS if q["id"] == "colour")
print(f"\nQuestion: {test['ask']}\nReference answer: {test['reference']}")

passages = rerank(test["ask"], hybrid(test["ask"]), keep=3)
retrieved = {"Text RAG": passages, "Multimodal RAG": passages + pictures(test["ask"])}

for name, sources in retrieved.items():
    precision, recall = context_precision(test["ask"], sources, test["reference"]), context_recall(test["reference"], sources)
    step(f"{name}: {len(sources)} sources")
    for n, (source, useful) in enumerate(zip(sources, precision["useful"]), start=1):
        print(f"    [{n}] {'useful ' if useful else 'not    '} {source['type']:8} page {source['page']}")
    for statement in recall["statements"]:
        print(f"    {'found    ' if statement['found'] else 'NOT FOUND'}  {statement['statement']}")
    print(f"    context precision = {precision['score']:.2f}   context recall = {recall['score']:.2f}")
