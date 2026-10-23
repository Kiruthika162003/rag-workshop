"""The grounded answer: what the model is told, and how it cites.

Every source goes in numbered, with its page and section. The rules say: cite a number after every
fact, read figures and tables from the picture, and say "I could not find it" rather than guess.

The code: prompt() and answer() in core/answer.py, the rules ANSWER in core/prompts.py

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
