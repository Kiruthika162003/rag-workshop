"""Cut pieces into chunks: fixed every N characters, or semantic, where the topic changes."""

import numpy as np


def fixed_chunks(text: str, size: int = 500) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def semantic_chunks(pieces: list[dict], vectors: list[np.ndarray], max_chars: int = 900) -> list[list[dict]]:
    """Start a new chunk at a heading, a row, a new page, a topic change, or when too long."""
    gaps = [float(a @ b) for a, b in zip(vectors, vectors[1:])]
    # "Not similar" is relative to this document: the lowest quarter of its neighbour similarities.
    cut_below = np.percentile(gaps, 25) if gaps else 0

    chunks, current, size = [], [], 0
    for i, piece in enumerate(pieces):
        starts_new = current and (
            piece["type"] in ("heading", "row")
            or current[-1]["type"] == "row"
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
