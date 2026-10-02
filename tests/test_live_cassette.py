"""Prove the Reader seam runs on the REAL recorded model, not only the stub.

The committed cassette `fixtures/cassettes/nimbus_reader.json` is a real recording from
`gemma3:27b` on a local Ollama server. These tests replay it through the full canonical
pipeline (bm25, the configured chunk_size/overlap and top_k) and check the generation
metrics. They use no network: the cassette is replayed by sha256(model, prompt).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ailab_rag.chunking import chunk_corpus
from ailab_rag.config import load_config
from ailab_rag.data import load_corpus, load_questions
from ailab_rag.eval import main
from ailab_rag.gen_checks import citation_resolvable, classify_refusal, lexical_support
from ailab_rag.providers import ReplayProvider
from ailab_rag.reader import Reader
from ailab_rag.retrieval import BM25
from tests.conftest import CASSETTE, CORPUS, QUESTIONS


def _replay_gen_checks() -> dict[str, object]:
    cfg = load_config()
    docs = load_corpus(cfg.corpus)
    questions = load_questions(cfg.questions)
    chunks = chunk_corpus(docs, cfg.chunk_size, cfg.chunk_overlap)
    text_by_id = {c.id: c.text for c in chunks}
    bm25 = BM25(chunks)
    reader = Reader(
        ReplayProvider(CASSETTE, name="ollama", model="gemma3:27b"),
        support_threshold=cfg.support_threshold,
    )
    buckets = {"answered": 0, "false_refusal": 0, "refusal_correct": 0, "wrong_refusal": 0}
    resolvable: list[float] = []
    support: list[float] = []
    for q in questions:
        retrieved = [(cid, text_by_id[cid]) for cid in bm25(q.question, cfg.top_k)]
        result = reader.answer(q.question, retrieved)
        buckets[classify_refusal(q.answerable, result.refused)] += 1
        if result.refused:
            continue
        resolvable.append(citation_resolvable(result.citations, [c for c, _ in retrieved]))
        support.append(
            max(
                (lexical_support(result.answer, text_by_id.get(c, "")) for c in result.citations),
                default=0.0,
            )
        )
    return {
        "buckets": buckets,
        "citation_resolvable": sum(resolvable) / len(resolvable),
        "lexical_support": sum(support) / len(support),
        "answered_n": len(resolvable),
    }


def test_cassette_is_a_real_gemma_recording() -> None:
    cass = json.loads(CASSETTE.read_text())
    assert cass["model"] == "gemma3:27b"
    assert len(cass["entries"]) == 18  # every golden question was recorded
    # A real recording follows the format but paraphrases, so support is high, not perfect.
    assert any(v["response"].strip().startswith("ANSWER:") for v in cass["entries"].values())


def test_real_reader_seam_gen_checks() -> None:
    checks = _replay_gen_checks()
    assert checks["buckets"] == {
        "answered": 14,
        "false_refusal": 0,
        "refusal_correct": 4,
        "wrong_refusal": 0,
    }
    assert checks["answered_n"] == 14
    # Every real citation resolves after the parser strips echoed brackets.
    assert float(checks["citation_resolvable"]) == pytest.approx(1.0)  # type: ignore[arg-type]
    # The real model paraphrases, so lexical support is strong but below the stub's 1.0.
    lexical = float(checks["lexical_support"])  # type: ignore[arg-type]
    assert 0.5 <= lexical < 1.0


def test_replay_mode_eval_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no eval_config.toml here, so built-in defaults apply
    out = tmp_path / "r.json"
    code = main(
        [
            "--provider", "replay", "--model", "gemma3:27b",
            "--cassette", str(CASSETTE), "--corpus", str(CORPUS),
            "--questions", str(QUESTIONS), "--output", str(out),
        ]
    )  # fmt: skip
    record = json.loads(out.read_text())
    assert code == 0
    assert (record["provider"], record["model"]) == ("ollama", "gemma3:27b")
    assert record["gen_checks"]["citation_resolvable"] == 1.0
    assert record["refusals"]["wrong_refusal"] == 0
