"""Two ways to cut documents into chunks.

fixed_chunks: every 500 characters, wherever that lands. Quick, and it splits sentences in half.
semantic_chunks: keep neighbouring sentences together while they talk about the same thing,
and start a new chunk where the topic changes. A chunk never crosses a page, so every chunk
has exactly one page number to cite.
"""

import numpy as np


def fixed_chunks(text: str, size: int = 500) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def semantic_chunks(pieces: list[dict], vectors: list[np.ndarray], max_chars: int = 900) -> list[list[dict]]:
    """A new chunk starts at a heading, at a page break, when a chunk gets too long, or where two
    neighbouring pieces stop being similar, once the chunk has enough to stand on its own."""
    gaps = [float(a @ b) for a, b in zip(vectors, vectors[1:])]
    # "Not similar" is relative to this document: the lowest quarter of its neighbour similarities.
    cut_below = np.percentile(gaps, 25) if gaps else 0

    chunks, current, size = [], [], 0
    for i, piece in enumerate(pieces):
        starts_new = current and (
            piece["type"] == "heading"
            or piece["page"] != current[-1]["page"]
            or (gaps[i - 1] < cut_below and size > 200)
            or size + len(piece["text"]) > max_chars
        )
        if starts_new:
            chunks.append(current)
            current, size = [], 0
        current.append(piece)
        size += len(piece["text"])
    if current:
        chunks.append(current)
    return chunks
