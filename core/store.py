"""A vector store small enough to read in one sitting: vectors in a numpy file, records in JSON.

Production systems use Vertex AI Vector Search, AlloyDB, pgvector or similar, but they all do
what search() does here: compare the question's vector with every stored vector, keep the
records that pass the metadata filter, and return the closest ones.
"""

import json
from pathlib import Path

import numpy as np

INDEX = Path(__file__).parents[1] / "index"


class Store:
    def __init__(self, name: str):
        self.name = name
        self.records = json.loads((INDEX / f"{name}.json").read_text(encoding="utf-8"))
        self.vectors = np.load(INDEX / f"{name}.npy")

    @staticmethod
    def save(name: str, records: list[dict], vectors: list[np.ndarray]):
        INDEX.mkdir(exist_ok=True)
        (INDEX / f"{name}.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
        np.save(INDEX / f"{name}.npy", np.stack(vectors))

    def search(self, question_vector: np.ndarray, k: int = 5, kinds=None, after=None, latest_only=False) -> list[dict]:
        # Vectors are unit length, so a dot product is the cosine similarity.
        scores = self.vectors @ question_vector
        allowed = [self.passes(record, kinds, after, latest_only) for record in self.records]
        order = [i for i in np.argsort(-scores) if allowed[i]]
        return [{**self.records[i], "score": round(float(scores[i]), 3)} for i in order[:k]]

    def passes(self, record: dict, kinds, after, latest_only) -> bool:
        """The metadata filter: decided before ranking, so a filtered-out chunk can never win."""
        if kinds and record["kind"] not in kinds:
            return False
        if after and record["date"] < after:
            return False
        if latest_only and record["date"] < self.newest_in_series(record["series"]):
            return False
        return True

    def newest_in_series(self, series: str) -> str:
        return max(r["date"] for r in self.records if r["series"] == series)
