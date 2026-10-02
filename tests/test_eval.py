import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ailab_rag.config import EvalConfig
from ailab_rag.eval import current_commit, evaluate, main
from tests.conftest import CORPUS, QUESTIONS


def _cfg(**kw: object) -> EvalConfig:
    base: dict[str, object] = {"corpus": CORPUS, "questions": QUESTIONS}
    base.update(kw)
    return EvalConfig(**base)  # type: ignore[arg-type]


def test_canonical_run_passes() -> None:
    result = evaluate(_cfg(min_recall=0.70, min_hit=0.70, min_mrr=0.50, min_ndcg=0.55))
    assert result["passed"] is True
    assert result["metrics"]["recall@5"] >= 0.9
    assert result["baselines"]["bm25_recall"] > result["baselines"]["random_recall"]
    assert result["refusals"]["wrong_refusal"] == 0
    assert result["refusals"]["refusal_correct"] == 4
    assert result["critical"]["n"] >= 1
    assert result["saturation"]["saturated"] is False


def test_results_record_contract() -> None:
    result = evaluate(_cfg())
    contract: dict[str, type | tuple[type, ...]] = {
        "schema_version": int,
        "lab": str,
        "dataset": str,
        "provider": str,
        "model": str,
        "primary_metric": str,
        "metrics": dict,
        "threshold": float,
        "passed": bool,
        "commit": (str, type(None)),
        "generated_at": str,
    }
    for key, kind in contract.items():
        assert isinstance(result[key], kind), key
    assert result["schema_version"] == 1
    assert result["lab"] == "ailab-rag"
    assert result["primary_metric"] == "recall@5"
    assert len(result["corpus_sha256"]) == 64
    assert datetime.fromisoformat(result["generated_at"]).utcoffset() == timedelta(0)


def test_random_retriever_regresses_recall() -> None:
    result = evaluate(_cfg(retriever="random", min_recall=0.70))
    assert result["passed"] is False
    assert any("recall@5" in f for f in result["failures"])


def test_saturation_guard_fires_on_easy_config(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        '{"id":"d1","title":"A","text":"alpha unique keyword answer here now today"}\n'
        '{"id":"d2","title":"B","text":"beta distinct keyword answer here now today"}\n'
    )
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        '{"id":"q1","question":"alpha","doc_id":"d1","answer_span":"alpha"}\n'
        '{"id":"q2","question":"beta","doc_id":"d2","answer_span":"beta"}\n'
    )
    result = evaluate(
        _cfg(corpus=corpus, questions=questions, chunk_size=16, chunk_overlap=0, top_k=2)
    )
    assert result["saturation"]["saturated"] is True
    assert any("saturation" in f for f in result["failures"])


def test_insufficient_class_is_reported(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id":"d1","title":"A","text":"alpha beta gamma answer span here now"}\n')
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"id":"q1","question":"alpha","doc_id":"d1","answer_span":"alpha"}\n')
    result = evaluate(_cfg(corpus=corpus, questions=questions, chunk_size=16, chunk_overlap=0))
    assert "trap" in result["classes"]["insufficient"]


def test_empty_corpus_raises(tmp_path: Path) -> None:
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    from ailab_rag.data import DatasetError

    with pytest.raises(DatasetError, match="empty"):
        evaluate(_cfg(corpus=empty))


def test_main_writes_results_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AILAB_COMMIT", "abc1234deadbeef")
    out = tmp_path / "r.json"
    code = main(
        [
            "--corpus", str(CORPUS), "--questions", str(QUESTIONS),
            "--output", str(out), "--min-recall", "0.5",
        ]
    )  # fmt: skip
    assert code == 0
    record = json.loads(out.read_text())
    assert record["commit"] == "abc1234deadbeef"
    stdout = capsys.readouterr().out
    assert "recall@k" in stdout
    assert "baselines:" in stdout


def test_main_bad_data_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--corpus", str(tmp_path / "nope.jsonl")]) == 2
    assert "error" in capsys.readouterr().err


def test_table_from_existing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "r.json"
    assert main(["--corpus", str(CORPUS), "--questions", str(QUESTIONS), "--output", str(out)]) == 0
    capsys.readouterr()
    assert main(["--table-from", str(out)]) == 0
    assert "| bm25 |" in capsys.readouterr().out
    (tmp_path / "bad.json").write_text("{}")
    assert main(["--table-from", str(tmp_path / "bad.json")]) == 2


def test_writes_github_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert (
        main(
            [
                "--corpus",
                str(CORPUS),
                "--questions",
                str(QUESTIONS),
                "--output",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    assert "### Eval results" in summary.read_text()


def test_commit_null_when_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert current_commit() is None
    monkeypatch.setenv("AILAB_COMMIT", "unknown")
    assert current_commit() is None
