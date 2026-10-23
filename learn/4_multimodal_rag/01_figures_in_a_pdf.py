"""Figures in a PDF: three different things that all look like "a picture" on the page.

    image     Figures 1 and 2 are pictures stored inside the PDF: extract them as they are
    drawing   Figures 3 to 5 are hundreds of vector lines and loose words: no image to extract,
              so the area they cover is rendered into a picture
    table     Tables 3 and 4 come out of the text layer with merged cells: rendered into a picture
              as well, so the model can read the grid the way a person does

This lesson writes every one of them to out/figures/, so you can open them and look.

The code: pieces() and drawn_figures() in core/documents.py

Run: python learn/4_multimodal_rag/01_figures_in_a_pdf.py
"""

from pathlib import Path

from core.documents import library, pieces
from core.show import step

out = Path("out/figures")
out.mkdir(parents=True, exist_ok=True)

step("Every figure and table, and how it was found:")
for n, piece in enumerate(p for p in pieces(library()[0]["doc"]) if p["type"] == "picture"):
    (out / f"p{n}.png").write_bytes(piece["data"])
    x0, y0, x1, y1 = piece["box"]
    print(f"    out/figures/p{n}.png   page {piece['page']:<3} {piece['from']:8} {int(x1 - x0)} x {int(y1 - y0)} points")
