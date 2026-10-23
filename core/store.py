"""The vector store: ChromaDB, on this computer, in index/chroma. Search and metadata filters run in Chroma."""

import json
import re
from functools import cache, cached_property
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from core.keywords import Keywords

INDEX = Path(__file__).parents[1] / "index"
LEVELS = ("s1", "s2", "s3")  # "5 Training", "5.2 Hardware and Schedule", ... one metadata field per level


@cache
def client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(INDEX / "chroma"), settings=Settings(anonymized_telemetry=False))


class Store:
    def __init__(self, name: str):
        self.collection = client().get_collection(name)
        got = self.collection.get(include=["embeddings", "metadatas"])
        order = sorted(range(len(got["ids"])), key=lambda i: got["metadatas"][i]["n"])
        self.records = [json.loads(got["metadatas"][i]["record"]) for i in order]
        self.vectors = np.array([got["embeddings"][i] for i in order], dtype=np.float32)  # for MMR and the metrics
        self.row = {record["id"]: i for i, record in enumerate(self.records)}

    @staticmethod
    def save(name: str, records: list[dict], vectors):
        if name in [c.name for c in client().list_collections()]:
            client().delete_collection(name)
        collection = client().create_collection(name, configuration={"hnsw": {"space": "cosine"}})
        parts = [record["section"].split(" / ") for record in records]
        collection.add(
            ids=[record["id"] for record in records],
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            documents=[record["text"] for record in records],
            metadatas=[{"n": n, "page": record["page"], "record": json.dumps(record),
                        **{level: (path[i] if i < len(path) else "") for i, level in enumerate(LEVELS)}}
                       for n, (record, path) in enumerate(zip(records, parts))],
        )

    @cached_property
    def keywords(self) -> Keywords:
        return Keywords([record["text"] for record in self.records])

    @cached_property
    def sections(self) -> set[str]:
        return {part for record in self.records for part in record["section"].split(" / ")}

    @staticmethod
    def matches(part: str, section: str) -> bool:
        """"5" matches "5 Training", "Training" matches "5 Training", "5.2" matches "5.2 Hardware ..."."""
        return bool(re.match(rf"(\d+(\.\d+)*\s)?{re.escape(section)}", part, re.I))

    @staticmethod
    def passes(record: dict, section=None, pages=None) -> bool:
        """The metadata filter, in Python: section, like "5.2" or "Training", and pages (first, last)."""
        in_section = not section or any(Store.matches(part, section) for part in record["section"].split(" / "))
        return in_section and (not pages or pages[0] <= record["page"] <= pages[1])

    def where(self, section=None, pages=None) -> dict | None:
        """The same filter, as a Chroma where clause."""
        clauses = []
        if section:
            names = [part for part in self.sections if self.matches(part, section)] or ["no such section"]
            clauses.append({"$or": [{level: {"$in": names}} for level in LEVELS]})
        if pages:
            clauses += [{"page": {"$gte": pages[0]}}, {"page": {"$lte": pages[1]}}]
        return None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, vector: np.ndarray, k: int = 5, **filters) -> list[dict]:
        """Vector search in Chroma, filtered before ranking. Chroma returns cosine distance; similarity = 1 - distance."""
        found = self.collection.query(query_embeddings=[vector.tolist()], n_results=min(k, len(self.records)),
                                      where=self.where(**filters), include=["distances"])
        return [{**self.records[self.row[ref]], "score": round(1 - distance, 3)}
                for ref, distance in zip(found["ids"][0], found["distances"][0])]

    def top(self, scores: np.ndarray, k: int = 5, **filters) -> list[dict]:
        """The k best by any other score, like BM25, with the same filter."""
        allowed = np.array([self.passes(record, **filters) for record in self.records])
        scores = np.where(allowed, scores, -np.inf)
        best = [i for i in np.argsort(-scores)[:k] if allowed[i]]
        return [{**self.records[i], "score": round(float(scores[i]), 3)} for i in best]


@cache
def store(name: str) -> Store:
    """Each index is loaded once and shared."""
    return Store(name)
