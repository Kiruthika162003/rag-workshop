"""Context precision and context recall: judge what was retrieved, not what was written.

    context precision   of the sources retrieved, how many are useful, and are the useful ones first?
                        For every useful source: the share of useful sources at or above its place,
                        averaged. 1.0 means every useful source came before every useless one.
    context recall      of the statements in the reference answer, how many can be found in the sources?
                        It needs a reference answer, written by a person.

The question is about a colour in Figure 1. Text RAG retrieves passages, so its recall is near 0: the
colour is not in any of them. Multimodal RAG retrieves the figure too, so it can reach 1.

The code: context_precision() and context_recall() in core/metrics.py, prompts USEFUL and ATTRIBUTED

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
