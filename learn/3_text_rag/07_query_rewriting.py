"""Query rewriting: search with better words than the ones the user typed.

Someone asks "how long did it take to train the big one?". The paper says "300,000 steps" and
"3.5 days". Two ways to close that gap:

    multi-query   the model writes three other phrasings; each is searched, and the lists are fused
    HyDE          the model writes a made-up answer, and we search with that: an answer looks
                  more like the passage we want than a question does (Hypothetical Document Embeddings)

The code: rewrites(), multi_query(), hyde() in core/query.py, prompts REWRITE and HYPOTHETICAL

Run: python learn/3_text_rag/07_query_rewriting.py
     python learn/3_text_rag/07_query_rewriting.py "does it work for things other than translation?"
"""

from core.query import hyde, hypothetical_answer, multi_query, rewrites
from core.search import by_meaning
from core.show import chunks, question, step

asked = question("how long did it take to train the big one?")

step("As typed:")
chunks(by_meaning(asked, k=3))

step("1. Multi-query. The model's other phrasings:")
for search in rewrites(asked):
    print("    -", search)
step("   Each one searched, the four lists fused:")
chunks(multi_query(asked, k=3))

step("2. HyDE. The made-up answer we search with (its facts do not matter, its wording does):")
print("   ", hypothetical_answer(asked))
chunks(hyde(asked, k=3))
