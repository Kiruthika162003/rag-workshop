"""Open the page: a link to the cited page, and a picture of it with the passage highlighted.

Say: A citation is only useful if you can check it. Every chunk knows its page and its sentences,
     so we open the PDF at that page, and draw the page with the passage highlighted.

Code: page_picture() in core/documents.py, /page/<ref> in app/app.py

Run: python learn/3_text_rag/12_open_the_page.py
     python learn/3_text_rag/12_open_the_page.py "What is multi-head attention?"
"""

from pathlib import Path

from core.documents import DOCS, page_picture
from core.rerank import rerank
from core.search import hybrid
from core.show import question, step

asked = question("What hardware did they train on?")
source = rerank(asked, hybrid(asked))[0]

step(f"The best source: page {source['page']}, {source['section']}, chunk {source['id']}")
print("    The sentences it was made from, to find on the page:")
for sentence in source["find"]:
    print("     -", sentence[:110])

step("A link that opens the file at that page:")
print("   ", f"{(DOCS / source['doc']).as_uri()}#page={source['page']}")

Path("out").mkdir(exist_ok=True)
Path("out/page.png").write_bytes(page_picture(source["doc"], source["page"], source["find"]))
step("The page with the passage highlighted: out/page.png")
