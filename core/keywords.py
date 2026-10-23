"""Keyword search with BM25, the scoring behind Elasticsearch, OpenSearch and most search boxes.

Embeddings match meaning, so "dff = 2048" and "dff = 4096" look almost the same to them.
Keyword search matches the exact words, and BM25 weighs each word by how rare it is:
a word in every chunk, like "model", counts for little; a word in one chunk, like "warmup", counts a lot.
"""

import math
import re
from collections import Counter

import numpy as np


# Words in nearly every question, which say nothing about which chunk is the answer.
STOP = set("a an and are at be by can do does for from has have how i if in is it its my of on or "
           "the this that to was what when where which who will with yet you your".split())


def words(text: str) -> list[str]:
    """Lower case words, without the stop words. Words like "dot-product" stay whole."""
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if w not in STOP]


class Keywords:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.counts = [Counter(words(text)) for text in texts]
        lengths = np.array([sum(count.values()) for count in self.counts])
        # k1: how fast repeating a word stops helping. b: how much long chunks are held back.
        self.k1, self.room = k1, k1 * (1 - b + b * lengths / lengths.mean())
        holding = Counter(word for count in self.counts for word in count)
        self.rarity = {word: math.log(1 + (len(texts) - n + 0.5) / (n + 0.5)) for word, n in holding.items()}

    def scores(self, query: str) -> np.ndarray:
        asked = set(words(query))
        return np.array([sum(self.rarity[w] * count[w] * (self.k1 + 1) / (count[w] + room) for w in asked & count.keys())
                         for count, room in zip(self.counts, self.room)])
