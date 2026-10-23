"""Self-query: let the model read the filters out of the question.

People do not type --section 5.3. They ask "in the training part, what optimizer did they use?".
One small Gemini call turns the question into a cleaner search and the filters it implies, then
the search runs with them.

The code: filters_from() in core/query.py, the prompt FILTERS in core/prompts.py

Run: python learn/3_text_rag/04_self_query.py
     python learn/3_text_rag/04_self_query.py "What do pages 13 to 15 show?"
"""

from core import prompts
from core.query import filters_from
from core.search import by_meaning
from core.show import chunks, question, step

asked = question("In the training part of the paper, what optimizer did they use?")

step("1. The prompt (FILTERS in core/prompts.py), up to the list of sections:")
print("    | " + prompts.FILTERS.split("Its top-level")[0].strip().replace("\n", "\n    | "))

search = filters_from(asked)
step("2. What the model read out of the question:")
for name, value in search.items():
    print(f"    {name:8} {value}")

step("3. The search, with those filters:")
chunks(by_meaning(search["query"], k=4, section=search["section"], pages=search["pages"]))
