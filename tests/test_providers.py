import json
from pathlib import Path

import pytest

from ailab_rag.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    StubProvider,
    cassette_key,
)
from ailab_rag.reader import Reader
from tests.conftest import CASSETTE


def test_stub_is_offline_and_picks_best_overlap_chunk() -> None:
    stub = StubProvider()
    prompt = (
        "Question: how often does the relay upload\n\n"
        "Context:\n[d#0] unrelated text about billing\n"
        "[d#1] the relay uploads changes every ninety seconds by default\n"
    )
    out = stub.complete(prompt)
    assert "CITE: d#1" in out
    assert out.startswith("ANSWER:")
    assert stub.calls == 1
    assert isinstance(stub, LLMProvider)


def test_stub_handles_empty_context() -> None:
    out = StubProvider().complete("Question: anything\n\nContext:\n")
    assert "cannot answer" in out.lower()


def test_cassette_key_is_stable() -> None:
    assert cassette_key("m", "p") == cassette_key("m", "p")
    assert cassette_key("m", "p") != cassette_key("m", "q")


def test_replay_provider_replays_real_cassette_on_canonical_pipeline() -> None:
    """The committed cassette is a REAL gemma3:27b recording. Replaying it must reproduce a
    cited answer for an answerable question, with the SAME prompt the recorder used."""
    from ailab_rag.chunking import chunk_corpus
    from ailab_rag.config import load_config
    from ailab_rag.data import load_corpus, load_questions
    from ailab_rag.retrieval import BM25

    cass = json.loads(CASSETTE.read_text())
    assert cass["model"] == "gemma3:27b"  # a real local model, not the stub
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
    q = next(q for q in questions if q.id == "q01")
    retrieved = [(cid, text_by_id[cid]) for cid in bm25(q.question, cfg.top_k)]
    result = reader.answer(q.question, retrieved)
    assert not result.refused
    assert result.citations  # a resolvable citation, brackets stripped by the parser
    assert all(c in text_by_id for c in result.citations)
    assert "ninety seconds" in result.answer.lower()


def test_replay_miss_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"model": "m", "entries": {}}))
    replay = ReplayProvider(path, model="m")
    with pytest.raises(CassetteMissError, match="cassette miss"):
        replay.complete("a prompt that was never recorded")


def test_replay_accepts_bare_mapping(tmp_path: Path) -> None:
    key = cassette_key("m", "p")
    path = tmp_path / "c.json"
    path.write_text(json.dumps({key: {"model": "m", "response": "ANSWER: hi\nCITE: x"}}))
    assert ReplayProvider(path, model="m").complete("p") == "ANSWER: hi\nCITE: x"


def test_ollama_provider_builds_localhost_url() -> None:
    p = OllamaProvider("llama3.2", host="localhost:11434")
    assert p.url == "http://localhost:11434/api/generate"
    assert p.model == "llama3.2"
    assert p.name == "ollama"
    assert isinstance(p, LLMProvider)
