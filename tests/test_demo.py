from pathlib import Path

import pytest

from ailab_rag.demo import main as demo_main
from tests.conftest import CORPUS, QUESTIONS


def test_demo_prints_stages_and_runs_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AILAB_CORPUS", str(CORPUS))
    monkeypatch.setenv("AILAB_QUESTIONS", str(QUESTIONS))
    code = demo_main(
        [
            "--corpus",
            str(CORPUS),
            "--questions",
            str(QUESTIONS),
            "--output",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "ailab-rag demo" in out
    assert "docs ->" in out
    assert "score=" in out  # retrieved chunk ids printed with scores
    assert "worked example" in out  # the excluded-chunk observation
    assert "== eval ==" in out
