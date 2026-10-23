"""Query rewriting: multi-query searches other phrasings; HyDE searches with a made-up answer.

Say: Users and papers use different words. Multi-query: the model writes other phrasings, each is
     searched, the lists are merged. HyDE: the model writes a made-up answer and we search with
     that, because an answer looks more like the passage than a question does.

Code: rewrites(), multi_query(), hyde() in core/query.py, prompts REWRITE and HYPOTHETICAL

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
