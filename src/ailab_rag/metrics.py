"""Retrieval metrics, ported VERBATIM from ``ailab-evals`` (see DESIGN.md for the SHA).

The four ranking functions below are copied character-for-character from
``ailab_evals.metrics`` so the maths is not re-derived. ``tests/test_metric_parity.py``
imports both these copies and the originals and asserts they agree on a shared fixture,
so the two cannot drift.

A helper maps a question to its relevant chunk-id set via span containment for a given
chunk set; retrieval metrics are then computed against that set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ailab_rag.chunking import Chunk, relevant_chunk_ids
from ailab_rag.data import Question

# --- ported verbatim from ailab-evals ------------------------------------------------


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


# --- lab helper ----------------------------------------------------------------------


def question_relevance(question: Question, chunks: list[Chunk]) -> set[str]:
    """Map one question to its relevant chunk-id set via span containment.

    A thin wrapper over :func:`ailab_rag.chunking.relevant_chunk_ids`, kept here so the
    retrieval metrics and their relevance source sit in one module.
    """
    return relevant_chunk_ids(question, chunks)


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence (so an empty class reads as such)."""
    return sum(values) / len(values) if values else None
