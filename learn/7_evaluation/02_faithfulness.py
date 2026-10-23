"""Faithfulness: the share of the answer's claims that the sources support.

Say: Is every claim in the answer backed by a source? A judge model checks each one. Add one
     invented sentence, and the score drops from 1.00 to 0.80: it is caught.

Code: faithfulness() in core/metrics.py, the prompt FAITHFUL in core/prompts.py

Run: python learn/7_evaluation/02_faithfulness.py
     python learn/7_evaluation/02_faithfulness.py "What optimizer was used?"
"""

from core.metrics import faithfulness
from core.rag import text
from core.show import question, step

asked = question("What regularization does the base model use?")
result = text(asked)
invented = " The authors also used weight decay of 0.01 on every layer."

for title, reply in (("The real answer:", result["answer"]), ("The same answer, with one invented sentence:", result["answer"] + invented)):
    step(title)
    print("   ", reply)
    judged = faithfulness(reply, result["sources"])
    for claim in judged["claims"]:
        print(f"    {'supported  ' if claim['supported'] else 'UNSUPPORTED'}  {claim['claim']}")
    print(f"    faithfulness = {judged['score']:.2f}")
