"""A vector store small enough to read in one sitting: vectors in a numpy file, records in JSON.

Production systems use Vertex AI Vector Search, AlloyDB, pgvector or similar, but they all do
what this file does: score every stored chunk, drop the ones the metadata filter rules out, and
return the best of what is left.
"""

import json
import re
from functools import cache, cached_property
from pathlib import Path

import numpy as np

from core.keywords import Keywords

INDEX = Path(__file__).parents[1] / "index"


class Store:
    def __init__(self, name: str):
        self.records = json.loads((INDEX / f"{name}.json").read_text(encoding="utf-8"))
        self.vectors = np.load(INDEX / f"{name}.npy")
        self.row = {record["id"]: i for i, record in enumerate(self.records)}

    @staticmethod
    def save(name: str, records: list[dict], vectors):
        INDEX.mkdir(exist_ok=True)
        (INDEX / f"{name}.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
        np.save(INDEX / f"{name}.npy", np.asarray(vectors, dtype=np.float32))

    @cached_property
    def keywords(self) -> Keywords:
        return Keywords([record["text"] for record in self.records])

    @staticmethod
    def passes(record: dict, section=None, pages=None) -> bool:
        """The metadata filter. section: a number or a name, like "5.2" or "Training", which also
        matches everything inside it. pages: (first, last)."""
        in_section = not section or re.search(rf"(^|/ )(\d+(\.\d+)*\s)?{re.escape(section)}", record["section"], re.I)
        on_pages = not pages or pages[0] <= record["page"] <= pages[1]
        return bool(in_section and on_pages)

    def top(self, scores: np.ndarray, k: int = 5, **filters) -> list[dict]:
        """The k best by any score. Filtered out before ranking, so a ruled-out chunk can never win."""
        allowed = np.array([self.passes(record, **filters) for record in self.records])
        scores = np.where(allowed, scores, -np.inf)
        best = [i for i in np.argsort(-scores)[:k] if allowed[i]]
        return [{**self.records[i], "score": round(float(scores[i]), 3)} for i in best]

    def search(self, vector: np.ndarray, k: int = 5, **filters) -> list[dict]:
        # Vectors are unit length, so a dot product is the cosine similarity.
        return self.top(self.vectors @ vector, k, **filters)


@cache
def store(name: str) -> Store:
    """Each index is loaded once and shared."""
    return Store(name)
