"""Take the reader to the page: a link that opens the PDF at the cited page, and the passage highlighted.

Every chunk remembers its file, its page and the exact sentences it was made from. A link with
#page=N opens most PDF viewers at that page. To show the passage, the page is drawn as a picture
and each of those sentences is found on it and highlighted, the way the app's source viewer does.

The code: page_picture() in core/documents.py, /page/<ref> in app/app.py

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
