"""Faithfulness: is every claim in the answer backed by a source?

A fluent answer can still add things the sources never said. A second model call splits the answer
into claims and checks each one against the sources it was written from. To show what a failure
looks like, the same check then runs on the answer with one invented sentence added to it.

The code: grounded() in core/answer.py, the prompt GROUNDED in core/prompts.py

Run: python learn/7_evaluation/02_faithfulness.py
     python learn/7_evaluation/02_faithfulness.py "What optimizer was used?"
"""

from core.answer import grounded
from core.rag import text
from core.show import question, step

asked = question("What regularization does the base model use?")
result = text(asked)
invented = " The authors also used weight decay of 0.01 on every layer."

for title, reply in (("The real answer:", result["answer"]), ("The same answer, with one invented sentence:", result["answer"] + invented)):
    step(title)
    print("   ", reply)
    for claim in grounded(reply, result["sources"]):
        print(f"    {'supported  ' if claim['supported'] else 'UNSUPPORTED'}  {claim['claim']}")
