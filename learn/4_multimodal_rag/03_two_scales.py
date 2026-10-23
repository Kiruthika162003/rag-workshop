"""Two scales: why picture scores and text scores cannot simply be compared.

Each figure is found two ways: by what it looks like (CLIP, picture against question) and by its
description (the text model, description against question). CLIP scores sit around 0.2 to 0.3,
text scores around 0.6 to 0.8: the gap between modalities. Taking the larger number would mean
the text score always wins. Ranks do not have that problem, so the two lists are fused by rank,
the same reciprocal rank fusion as hybrid search.

The code: pictures() in core/search.py

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
