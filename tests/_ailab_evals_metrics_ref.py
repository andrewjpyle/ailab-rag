"""Frozen VERBATIM copy of the four retrieval metrics from ailab-evals.

Source: ailab-evals @ 8c3ff14597b462fd927e1ac0c16f66c09392c586, src/ailab_evals/metrics.py.
This file exists so tests/test_metric_parity.py can assert the ported copies in
ailab_rag.metrics agree with the originals even in CI, where the ailab-evals clone is not
present. When the clone IS present, the parity test also checks this frozen copy still
matches the live source, so it cannot silently drift from the pinned SHA.

Do NOT edit by hand: re-copy from the pinned source if the originals ever change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


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
