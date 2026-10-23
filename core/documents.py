"""Read every file into pieces a search index can use, and remember where each piece came from."""

import json
import re
from datetime import datetime
from pathlib import Path

import pymupdf

pymupdf.no_recommend_layout()
DOCS = Path(__file__).parents[1] / "data" / "docs"
MARGIN = 50  # points at the top and bottom of a page: the header and footer repeated on every page
MEDIA = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4"}
PLAIN = pymupdf.TEXTFLAGS_BLOCKS & ~pymupdf.TEXT_PRESERVE_LIGATURES  # "ﬂ" comes out as "fl", so search can match it


def library() -> list[dict]:
    """Every file with its metadata. PDFs carry their own; audio and video have a JSON file beside them."""
    files = []
    for path in sorted(DOCS.iterdir()):
        if path.suffix == ".pdf":
            info = pymupdf.open(path).metadata
            meta = {"title": info["title"], "kind": info["subject"],
                    "date": datetime.strptime(info["creationDate"][2:10], "%Y%m%d").date().isoformat()}
        elif path.suffix in MEDIA:
            meta = json.loads(path.with_suffix(".json").read_text())
        else:
            continue
        # Two versions of one document share a series: "Refund Policy (February 2024)" and "(January 2026)".
        files.append({"doc": path.name, **meta, "series": re.sub(r"\s*\(.*\)$", "", meta["title"])})
    return files


def raw_text(doc: str) -> list[tuple[int, str]]:
    """What a naive pipeline sees: the text layer of each page, headers and footers included."""
    if not doc.endswith(".pdf"):
        return []
    return [(page.number + 1, page.get_text()) for page in pymupdf.open(DOCS / doc)]


def pieces(doc: str) -> list[dict]:
    """Paragraphs become sentences, table rows become sentences that keep their column names,
    and pictures, audio and video are kept as they are. Headers and footers are dropped."""
    path = DOCS / doc
    if path.suffix in MEDIA:
        return [{"page": 1, "type": MEDIA[path.suffix].split("/")[0], "mime": MEDIA[path.suffix], "data": path.read_bytes()}]

    found = []
    pdf = pymupdf.open(path)
    for page in pdf:
        number, on_page = page.number + 1, []
        tables = page.find_tables().tables
        for table in tables:
            on_page += table_rows(page, table)
        for x0, y0, x1, y1, text, _, kind in page.get_text("blocks", flags=PLAIN):
            text = " ".join(text.split())
            box = pymupdf.Rect(x0, y0, x1, y1)
            if kind or not text or y1 < MARGIN or y0 > page.rect.height - MARGIN or any(box.intersects(t.bbox) for t in tables):
                continue
            if len(text) < 60 and not text.endswith("."):
                on_page.append({"page": number, "y": y0, "type": "heading", "text": text, "find": [text]})
            else:
                on_page += [{"page": number, "y": y0, "type": "sentence", "text": s, "find": [s]}
                            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)]
        for xref, *_ in page.get_images(full=True):
            picture = pymupdf.Pixmap(pdf, xref)
            if picture.n - picture.alpha > 3:
                picture = pymupdf.Pixmap(pymupdf.csRGB, picture)
            box = page.get_image_rects(xref)[0]
            on_page.append({"page": number, "y": box.y0, "type": "picture", "mime": "image/png",
                            "data": picture.tobytes("png"), "box": list(box)})
        found += sorted(on_page, key=lambda piece: piece["y"])  # reading order: top of the page to the bottom
    return found


def table_rows(page, table) -> list[dict]:
    """One sentence per row that keeps the column names, so 'Monument Square' stays next to 'Priya Raman'."""
    words = page.get_textpage(flags=PLAIN)

    def read(row):
        return [" ".join(page.get_textbox(cell, textpage=words).split()) if cell else "" for cell in row.cells]

    header, *body = table.rows
    names = read(header)
    rows = []
    for row in body:
        values = read(row)
        rows.append({"page": page.number + 1, "y": row.bbox[1], "type": "row",
                     "text": "; ".join(f"{name}: {value}" for name, value in zip(names, values) if value),
                     "find": [value for value in values if value]})
    return rows


def page_picture(doc: str, page: int, find=(), box=None, zoom: float = 1.6) -> bytes:
    """One page as a PNG, with the passage that was used highlighted."""
    sheet = pymupdf.open(DOCS / doc)[page - 1]
    for text in find:
        for hit in sheet.search_for(text):
            sheet.add_highlight_annot(hit)
    if box:
        sheet.draw_rect(pymupdf.Rect(box), color=(0.98, 0.67, 0), width=3)
    return sheet.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
