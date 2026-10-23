"""Two scales: picture scores sit near 0.3, text near 0.7, so the two searches are merged by rank.

Say: Each figure is found two ways: by how it looks, and by its description. But picture scores
     sit near 0.3 and text scores near 0.75, so we cannot compare them. We merge by rank instead.

Code: pictures() in core/search.py

Run: python learn/4_multimodal_rag/03_two_scales.py
     python learn/4_multimodal_rag/03_two_scales.py "Which words does 'its' attend to?"
"""

from core.embeddings import embed_query, embed_query_for_pictures
from core.search import pictures
from core.show import question, step
from core.store import store

asked = question("the steps inside scaled dot-product attention")
looks = store("pictures").vectors @ embed_query_for_pictures(asked)
described = store("descriptions").vectors @ embed_query(asked)

step(f"    {'looks':>6}  {'described':>9}  figure")
for record, a, b in zip(store("pictures").records, looks, described):
    print(f"    {a:6.3f}  {b:9.3f}  {record['id']}  page {record['page']:<3} {record['text'].split('.')[0][:60]}")

step("Fused by rank, the two best:")
for figure in pictures(asked):
    print(f"    {figure['score']:.4f}  {figure['id']}  page {figure['page']}")
