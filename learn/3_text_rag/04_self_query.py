"""Self-query: the model reads the filters out of the question.

Say: Nobody types --section 5. They ask "in the training part, what optimizer?". One small model
     call turns that into a cleaner search plus the filter, then the search runs with it.

Code: filters_from() in core/query.py, the prompt FILTERS in core/prompts.py

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
