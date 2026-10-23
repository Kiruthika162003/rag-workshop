"""The grounded answer: the exact prompt, numbered sources with pages, and the cited answer.

Say: This is the whole message the model gets: numbered sources with their pages, and the rules.
     Cite every fact, say "could not find" rather than guess. Every fact traces back to a page;
     fine-tuning cannot give you that.

Code: prompt() and answer() in core/answer.py, the rules ANSWER in core/prompts.py

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
