"""Answer relevancy: does the answer address the question that was asked?

Say: Does the answer address the question? The judge writes the questions the answer would answer
     and compares them with the real one. "Could not find" is honest, but it scores 0 here.

Code: answer_relevancy() in core/metrics.py, the prompt ASKED in core/prompts.py

Run: python learn/7_evaluation/03_answer_relevancy.py
"""

from core.metrics import answer_relevancy
from core.show import question, step

asked = question("On what hardware was the big model trained, and for how long?")
answers = {
    "on point": "The big model was trained on one machine with 8 NVIDIA P100 GPUs, for 3.5 days.",
    "off the point": "The Transformer uses multi-head attention with 16 heads in its big configuration.",
    "gave up": "I could not find the answer in the provided sources.",
}
for name, answer in answers.items():
    judged = answer_relevancy(asked, answer)
    step(f"{name}: {answer}")
    for written in judged["questions"]:
        print("    would answer:", written)
    print(f"    answer relevancy = {judged['score']:.2f}")
