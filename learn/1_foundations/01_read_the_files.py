"""What a program sees in a PDF: the naive reader gets only text; the careful one also finds figures and tables.

Run: python learn/1_foundations/01_read_the_files.py
"""

from collections import Counter

from core.documents import library, pieces, raw_text
from core.show import step

PAPER = library()[0]
pages = raw_text(PAPER["doc"])
found = pieces(PAPER["doc"])

step(f"{PAPER['title']} ({PAPER['date'][:4]}), {len(pages)} pages.")
print(f"    A naive reader gets {sum(len(text) for _, text in pages):,} characters of text and nothing else.")
print(f"    A careful reader finds: {dict(Counter(p['type'] for p in found))}")

step("Page 3, as a naive reader sees it. Figure 1 is on this page, and it is not in the text:")
print("   ", " ".join(pages[2][1].split())[:300], "...")

step("Page 13, the first attention plot, as a naive reader sees it: words with no lines between them.")
print("   ", " ".join(pages[12][1].split())[:200], "...")

step("Where the figures are, and how each was found:")
for piece in found:
    if piece["type"] == "picture":
        how = {"image": "an image stored in the PDF", "drawing": "vector lines, rendered into a picture",
               "table": "a table, rendered into a picture"}[piece["from"]]
        print(f"    page {piece['page']:<3} {len(piece['data']) // 1024:>4} KB   {how}")

step("A table row, read as text that keeps its column names:")
print("   ", next(p["text"] for p in found if p["type"] == "row")[:160])

step("Headings, which become the section of every chunk under them:")
print("   ", " | ".join(p["text"] for p in found if p["type"] == "heading")[:400])
