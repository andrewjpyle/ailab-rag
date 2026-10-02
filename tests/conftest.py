from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "fixtures" / "nimbus_corpus.jsonl"
QUESTIONS = REPO_ROOT / "fixtures" / "nimbus_questions.jsonl"
CASSETTE = REPO_ROOT / "fixtures" / "cassettes" / "nimbus_reader.json"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host env vars from leaking into config-sensitive tests."""
    for var in (
        "AILAB_CONFIG",
        "AILAB_CORPUS",
        "AILAB_QUESTIONS",
        "AILAB_RETRIEVER",
        "AILAB_CHUNK_SIZE",
        "AILAB_CHUNK_OVERLAP",
        "AILAB_TOP_K",
        "AILAB_OUTPUT",
        "AILAB_MIN_RECALL",
        "AILAB_MIN_HIT",
        "AILAB_MIN_MRR",
        "AILAB_MIN_NDCG",
        "AILAB_TOLERANCE",
        "AILAB_PRIMARY_METRIC",
        "AILAB_SUPPORT_THRESHOLD",
        "AILAB_COMMIT",
        "GITHUB_SHA",
        "GITHUB_STEP_SUMMARY",
    ):
        monkeypatch.delenv(var, raising=False)
