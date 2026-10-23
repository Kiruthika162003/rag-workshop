"""The grounded answer: the exact prompt, numbered sources with pages, and the cited answer.

Run: python learn/3_text_rag/11_grounded_answer.py
     python learn/3_text_rag/11_grounded_answer.py "How many GPUs did they use?"
"""

from core.answer import answer, prompt
from core.rerank import rerank
from core.search import hybrid, with_neighbours
from core.show import prompt as show_prompt, question, sources, step

asked = question("What regularization does the base model use?")
best = with_neighbours(rerank(asked, hybrid(asked)))

step("The whole message the model gets:")
show_prompt(prompt(asked, best))

sources(answer(asked, best))
