"""Corrective RAG: check what came back before trusting it, and try again when it is weak.

A fixed pipeline answers from whatever the search returned, even when nothing there is any good.
Corrective RAG checks twice:

    1. the grade    the reranker's best score must be at least 7 out of 10
    2. the answer   it must not say it could not find the answer: a grader can be fooled by a
                    passage that only looks right

If either check fails, it writes other phrasings of the question, searches again, adds the figures
and tables, and answers from that. The optional step of scaled dot-product attention is named only
in Figure 2, so the first, text-only search cannot answer.

The code: corrective() in core/rag.py

Run: python learn/6_agentic_rag/01_corrective_rag.py
     python learn/6_agentic_rag/01_corrective_rag.py "How many warmup steps did the learning rate schedule use?"
"""

from core.rag import GOOD_ENOUGH, corrective
from core.show import question, sources, step

asked = question("In the scaled dot-product attention diagram, which step is optional, and what comes right after it?")
result = corrective(asked)

attempts = result["attempts"]
for n, attempt in enumerate(attempts, start=1):
    if n == len(attempts):
        verdict = "answer from this"
    elif attempt["best"] < GOOD_ENOUGH:
        verdict = "too weak, search again"
    else:
        verdict = "the grade passed, but the answer could not find it: search again"
    step(f"Attempt {n}: best passage scores {attempt['best']}/10, {verdict}")
    print("    searched: " + " | ".join(attempt["search"]))

sources(result)
