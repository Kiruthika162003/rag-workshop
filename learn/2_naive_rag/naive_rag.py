"""Naive RAG: the version in most tutorials, and where it goes wrong.

Chop the text into 500-character pieces, find the three most similar to the question, paste them
into a prompt. No sections, no figures, no tables it can read.

The code: naive() in core/rag.py, the prompt NAIVE in core/prompts.py

Run: python learn/2_naive_rag/naive_rag.py
     python learn/2_naive_rag/naive_rag.py "According to Table 3, how many parameters does the big model have?"
"""

from core import prompts
from core.gemini import ask
from core.search import by_meaning
from core.show import prompt, question, step

asked = question("In the scaled dot-product attention diagram, which step is optional, and what comes right after it?")

found = by_meaning(asked, k=3, index="naive")
step("1. The three most similar 500-character chunks. Look where they start and stop:")
for chunk in found:
    print(f"    {chunk['score']:.3f}  page {chunk['page']}:  |{' '.join(chunk['text'].split())[:90]}...")

message = prompts.NAIVE.format(context="\n\n".join(chunk["text"] for chunk in found), question=asked)
step("2. The whole prompt the model gets:")
prompt(message)

step("3. Answer:")
print("   ", ask(message))
print("\nThe answer is in Figure 2, a picture. Naive RAG never sees a picture.")
