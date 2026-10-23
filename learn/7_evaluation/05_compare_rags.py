"""All four metrics, for every kind of RAG, on one question: a small evaluation report.

    faithfulness       the answer sticks to its sources              catches made-up claims
    answer relevancy   the answer addresses the question             catches dodges and "could not find"
    context precision  useful sources were retrieved, ranked first   catches noisy retrieval
    context recall     the sources hold the whole right answer       catches missing evidence

Read the four together. A RAG that says "could not find" can be perfectly faithful and still useless;
a RAG with high recall and low faithfulness found the evidence and then wandered from it.

Cost: one answer and four judge calls per kind, about fifteen small calls.

The code: judge() in core/metrics.py

Run: python learn/7_evaluation/05_compare_rags.py               the figure question
     python learn/7_evaluation/05_compare_rags.py chain         any question id from core/questions.py
"""

import sys

from core import rag
from core.metrics import judge
from core.questions import QUESTIONS, correct
from core.show import step

test = next(q for q in QUESTIONS if q["id"] == (sys.argv[1] if len(sys.argv) > 1 else "colour"))
print(f"\nQuestion: {test['ask']}")

step(f"{'':16}{'right':>7}{'faithful':>10}{'relevant':>10}{'precision':>11}{'recall':>8}")
for name, run in (("Text RAG", rag.text), ("Multimodal RAG", rag.multimodal), ("Graph RAG", rag.graph_rag)):
    result = run(test["ask"])
    scores = judge(test["ask"], result["answer"], result["sources"], test["reference"])
    marks = [scores[m]["score"] for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")]
    print(f"{name:16}{'yes' if correct(test, result['answer']) else 'no':>7}" + "".join(f"{m:>10.2f}" for m in marks))
