"""Figures in a PDF: images, vector drawings and tables, each saved as a picture in out/figures.

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
