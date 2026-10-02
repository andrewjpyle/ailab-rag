"""Proof-of-gate: the gate must FAIL on each sabotage, naming the failing metric.

This mirrors ``make red``. Each case asserts the exit code and that the failing metric is
named in stderr, so a future change that quietly weakens the gate is caught.
"""

from pathlib import Path

import pytest

from ailab_rag.eval import main
from tests.conftest import CORPUS, QUESTIONS

BASE = ["--corpus", str(CORPUS), "--questions", str(QUESTIONS)]


def _run(args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    monkeypatch.chdir(tmp_path)
    code = main([*BASE, "--output", str(tmp_path / "r.json"), *args])
    return code, ""


def test_red_1_random_retriever_regresses_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(["--retriever", "random", "--min-recall", "0.7"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "REGRESSION: recall@5" in err


def test_red_1b_top_k_1_regresses_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(["--top-k", "1", "--min-recall", "0.7"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "recall@1" in err


def test_red_2_empty_data_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    code = main(["--corpus", str(empty), "--questions", str(QUESTIONS)])
    assert code == 2
    assert "empty" in capsys.readouterr().err


def test_red_3_force_answer_triggers_wrong_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(["--force-answer", "--min-recall", "0.7"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "REGRESSION: wrong_refusal" in err


def test_red_4_sabotage_citations_fails_lexical_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(["--sabotage-citations", "--min-recall", "0.7"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "REGRESSION: lexical_support" in err
