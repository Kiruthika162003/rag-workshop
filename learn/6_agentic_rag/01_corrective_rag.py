"""Corrective RAG: grade the search and check the answer; if either is weak, search again wider.

Say: Check before trusting the search. If the best passage scores under 7, or the answer says
     "could not find", search again with other words and add the figures.

Code: corrective() in core/rag.py

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
