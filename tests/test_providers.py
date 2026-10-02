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


def test_replay_provider_replays_committed_cassette() -> None:
    cass = json.loads(CASSETTE.read_text())
    model = cass["model"]
    replay = ReplayProvider(CASSETTE, name="ollama", model=model)
    reader = Reader(replay)
    retrieved = [
        (
            "nimbus-overview#1",
            "The Relay batches changes and uploads them every ninety seconds by default, "
            "which keeps network use low on slow links.",
        ),
        ("nimbus-overview#0", "Nimbus is a cloud backup and file sync service for small teams."),
    ]
    result = reader.answer("How often does the Relay upload changes by default?", retrieved)
    assert not result.refused
    assert result.citations == ("nimbus-overview#1",)
    assert "ninety seconds" in result.answer


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
