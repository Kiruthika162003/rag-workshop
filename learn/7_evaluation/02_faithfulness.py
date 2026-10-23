"""Faithfulness: is every claim in the answer backed by a source?

A fluent answer can still add things the sources never said. A judge model splits the answer into
claims and checks each one against the sources it was written from, figures included as pictures.

    faithfulness = supported claims / all claims

To show what a failure looks like, the same check then runs on the answer with one invented
sentence added to it. Two judge calls, plus the answer.

The code: faithfulness() in core/metrics.py, the prompt FAITHFUL in core/prompts.py

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
