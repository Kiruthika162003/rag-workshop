"""Read the paper into pieces a search index can use, and remember where each piece came from.

Three kinds of thing are on its pages:
    text       headings, sentences, and table rows that keep their column names
    pictures   figures stored as images (the architecture diagrams)
    drawings   figures drawn as vector lines (the attention plots): no image to extract,
               so the area is rendered into a picture
    tables     kept twice: as rows of text, and as a picture of the table
"""

import json
import re
from pathlib import Path

import pymupdf

pymupdf.no_recommend_layout()
DOCS = Path(__file__).parents[1] / "data" / "docs"
FOOTER = 60  # points at the bottom of a page: the page number
STAMP = 60   # points at the left edge: arXiv prints its identifier sideways there
PLAIN = pymupdf.TEXTFLAGS_BLOCKS & ~pymupdf.TEXT_PRESERVE_LIGATURES  # "ﬁ" comes out as "fi", so search can match it
HEADING = re.compile(r"^(\d+(\.\d+)*\s+)?[A-Z][A-Za-z -]+$")  # "3.2.1 Scaled Dot-Product Attention", "Abstract"


def library() -> list[dict]:
    """Every PDF with its metadata, from the JSON file beside it."""
    return [{"doc": path.name, **json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))}
            for path in sorted(DOCS.glob("*.pdf"))]


def raw_text(doc: str) -> list[tuple[int, str]]:
    """What a naive pipeline sees: the text layer of each page, page numbers and stamps included."""
    return [(page.number + 1, page.get_text()) for page in pymupdf.open(DOCS / doc)]


def drawn_figures(page) -> list[pymupdf.Rect]:
    """Areas covered by many vector lines: figures with no image inside them."""
    if len(page.get_drawings()) < 100:
        return []
    return [box for box in page.cluster_drawings() if box.width * box.height > 20_000]


def pieces(doc: str) -> list[dict]:
    """Headings, sentences, table rows and figures, in reading order, each with its page."""
    pdf = pymupdf.open(DOCS / doc)
    found = []
    for page in pdf:
        number, on_page = page.number + 1, []
        figures = drawn_figures(page)
        for box in figures:
            on_page.append({"page": number, "y": box.y0, "type": "picture", "from": "drawing", "mime": "image/png",
                            "box": list(box), "data": page.get_pixmap(clip=box, matrix=pymupdf.Matrix(2, 2)).tobytes("png")})
        tables = [t for t in page.find_tables().tables if not any(pymupdf.Rect(t.bbox).intersects(f) for f in figures)]
        for table in tables:
            # Read as text, a table with merged cells comes out scrambled. Keep the rows for keyword
            # search, and the table as a picture, so the model can read it the way a person does.
            on_page += table_rows(page, table)
            on_page.append({"page": number, "y": table.bbox[1], "type": "picture", "from": "table", "mime": "image/png",
                            "box": list(table.bbox),
                            "data": page.get_pixmap(clip=table.bbox, matrix=pymupdf.Matrix(2, 2)).tobytes("png")})
        tabled = [pymupdf.Rect(t.bbox) for t in tables]
        for x0, y0, x1, y1, text, _, kind in page.get_text("blocks", flags=PLAIN):
            text = " ".join(text.split())
            box = pymupdf.Rect(x0, y0, x1, y1)
            # Words inside a drawn figure are its labels, not sentences; a heading that only touches its edge is kept.
            inside = any(box in f for f in figures) or any(box.intersects(t) for t in tabled)
            if kind or len(text) < 4 or x1 < STAMP or y0 > page.rect.height - FOOTER or inside:
                continue
            if len(text) < 60 and HEADING.match(text):
                on_page.append({"page": number, "y": y0, "type": "heading", "text": text, "find": [text]})
            else:
                on_page += [{"page": number, "y": y0, "type": "sentence", "text": s, "find": [s]}
                            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)]
        for xref, *_ in page.get_images(full=True):
            picture = pymupdf.Pixmap(pdf, xref)
            if picture.n - picture.alpha > 3:
                picture = pymupdf.Pixmap(pymupdf.csRGB, picture)
            box = page.get_image_rects(xref)[0]
            on_page.append({"page": number, "y": box.y0, "type": "picture", "from": "image", "mime": "image/png",
                            "data": picture.tobytes("png"), "box": list(box)})
        found += sorted(on_page, key=lambda piece: piece["y"])  # reading order: top of the page to the bottom
    return found


def table_rows(page, table) -> list[dict]:
    """One sentence per row that keeps the column names, so "base" stays next to "dff: 2048"."""
    words = page.get_textpage(flags=PLAIN)

    def read(row):
        return [" ".join(page.get_textbox(cell, textpage=words).split()) if cell else "" for cell in row.cells]

    header, *body = table.rows
    names = read(header)
    rows = []
    for row in body:
        values = read(row)
        rows.append({"page": page.number + 1, "y": row.bbox[1], "type": "row",
                     "text": "; ".join(f"{name}: {value}" if name else value for name, value in zip(names, values) if value),
                     "find": [value for value in values if value]})
    return rows


def page_picture(doc: str, page: int, find=(), box=None, zoom: float = 1.6) -> bytes:
    """One page as a PNG, with the passage that was used highlighted and the figure outlined."""
    sheet = pymupdf.open(DOCS / doc)[page - 1]
    for text in find:
        for hit in sheet.search_for(text):
            sheet.add_highlight_annot(hit)
    if box:
        sheet.draw_rect(pymupdf.Rect(box), color=(0.98, 0.67, 0), width=3)
    return sheet.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
