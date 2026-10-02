"""Retrievers: a retriever is any ``fn(query, k) -> ranked list of chunk ids``.

``BM25`` is ported from ``ailab-evals`` (Okapi BM25, k1=1.5, b=0.75) and runs over the
chunk texts. Two baselines anchor the table and MUST score below BM25: ``RandomRetriever``
(seeded shuffle) and ``LeadRetriever`` (the first chunk of each document, score 0). The
baselines are deliberately weak, because a retrieval number means nothing without a floor
that a dumb retriever cannot clear.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from random import Random
from typing import Protocol, runtime_checkable

from ailab_rag.chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@runtime_checkable
class Retriever(Protocol):
    """Anything that ranks chunk ids for a query.

    ``name`` is recorded in the results. Keeping the seam this narrow is what lets BM25,
    a baseline, and a future dense retriever be scored by identical code.
    """

    name: str

    def __call__(self, query: str, k: int) -> list[str]: ...


class BM25:
    """Small Okapi BM25 baseline (k1=1.5, b=0.75). Ported from ailab-evals.retrieval."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        docs = {chunk.id: chunk.text for chunk in chunks}
        self.ids = list(docs)
        self.toks = [tokenize(docs[i]) for i in self.ids]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tf = [Counter(doc) for doc in self.toks]

    def scored(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(chunk_id, score)`` pairs with a positive score."""
        q = tokenize(query)
        scores = []
        for i, (tf, doc) in enumerate(zip(self.tf, self.toks, strict=True)):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += (
                        self.idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
                    )
            scores.append((s, self.ids[i]))
        scores.sort(key=lambda t: (-t[0], t[1]))
        return [(chunk_id, s) for s, chunk_id in scores[:k] if s > 0]

    def __call__(self, query: str, k: int) -> list[str]:
        return [chunk_id for chunk_id, _ in self.scored(query, k)]


class RandomRetriever:
    """Seeded random ranking of all chunk ids: a genuinely weak baseline."""

    name = "random"

    def __init__(self, chunks: Sequence[Chunk], seed: int = 0) -> None:
        self.ids = [chunk.id for chunk in chunks]
        self.seed = seed

    def __call__(self, query: str, k: int) -> list[str]:
        # Seed with the query so the ranking is deterministic per query, yet unrelated to
        # relevance. zlib-free: a simple stable hash of the query string.
        rng = Random(f"{self.seed}:{query}")
        order = list(self.ids)
        rng.shuffle(order)
        return order[:k]


class LeadRetriever:
    """Always returns the first chunk (index 0) of each document, score 0.

    Weak by design: the answer usually lives in a later chunk, so a retriever that only
    ever reads the opening of each document should lose to BM25.
    """

    name = "lead"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        # Preserve document order; keep only the lead chunk of each document.
        self.lead_ids = [chunk.id for chunk in chunks if chunk.index == 0]

    def __call__(self, query: str, k: int) -> list[str]:
        return self.lead_ids[:k]


def build_retriever(name: str, chunks: Sequence[Chunk], seed: int = 0) -> Retriever:
    """Factory keyed by config name. Add new retriever families here."""
    if name == "bm25":
        return BM25(chunks)
    if name == "random":
        return RandomRetriever(chunks, seed=seed)
    if name == "lead":
        return LeadRetriever(chunks)
    raise ValueError(f"unknown retriever {name!r}; choose one of bm25, random, lead")
