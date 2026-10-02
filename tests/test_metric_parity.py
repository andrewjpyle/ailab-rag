"""Parity test: the ported retrieval metrics must agree with the ailab-evals originals.

The port in ``ailab_rag.metrics`` is a verbatim copy. This test imports BOTH the port and
the originals and asserts they produce identical numbers on a shared fixture, so the two
cannot drift.

The originals are loaded from the live ailab-evals clone when ``$AILAB_EVALS_METRICS``
points at it, and otherwise from the frozen verbatim copy in
``tests/_ailab_evals_metrics_ref.py``. When the live clone IS present we also assert the
frozen copy still matches it, so the frozen copy cannot silently drift.
"""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from types import ModuleType

import pytest

from ailab_rag import metrics as ported
from tests import _ailab_evals_metrics_ref as frozen


def _load_live() -> ModuleType | None:
    env = os.environ.get("AILAB_EVALS_METRICS")
    if not env:
        return None
    path = Path(env)
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ailab_evals_metrics_live", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> list[tuple[list[str], set[str], int]]:
    rng = random.Random(1234)
    pool = [f"c{i}" for i in range(12)]
    cases: list[tuple[list[str], set[str], int]] = []
    for _ in range(50):
        ranked = rng.sample(pool, rng.randint(0, len(pool)))
        relevant = set(rng.sample(pool, rng.randint(0, 4)))
        k = rng.randint(1, 8)
        cases.append((ranked, relevant, k))
    cases.append(([], set(), 3))  # empty everything
    cases.append((["c0", "c1"], set(), 2))  # no relevant docs
    return cases


def _assert_agrees(reference: ModuleType) -> None:
    for ranked, relevant, k in _fixture():
        assert ported.recall_at_k(ranked, relevant, k) == reference.recall_at_k(ranked, relevant, k)
        assert ported.hit_at_k(ranked, relevant, k) == reference.hit_at_k(ranked, relevant, k)
        assert ported.reciprocal_rank(ranked, relevant) == reference.reciprocal_rank(
            ranked, relevant
        )
        assert ported.ndcg_at_k(ranked, relevant, k) == reference.ndcg_at_k(ranked, relevant, k)


def test_port_matches_frozen_reference() -> None:
    _assert_agrees(frozen)


def test_port_matches_live_clone_when_available() -> None:
    live = _load_live()
    if live is None:
        pytest.skip("ailab-evals clone not present; frozen-reference parity already asserted")
    _assert_agrees(live)
    # The frozen copy must still match the live source, so it cannot drift from the SHA.
    _assert_agrees_pair(frozen, live)


def _assert_agrees_pair(a: ModuleType, b: ModuleType) -> None:
    for ranked, relevant, k in _fixture():
        assert a.recall_at_k(ranked, relevant, k) == b.recall_at_k(ranked, relevant, k)
        assert a.ndcg_at_k(ranked, relevant, k) == b.ndcg_at_k(ranked, relevant, k)
