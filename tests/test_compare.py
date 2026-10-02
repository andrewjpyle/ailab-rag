import json
from pathlib import Path
from typing import Any

from ailab_rag.compare import changed_variables, main, metric_deltas, render


def _record(chunk_size: int, recall: float) -> dict[str, Any]:
    return {
        "retriever": "bm25",
        "passed": recall >= 0.9,
        "config": {"chunk_size": chunk_size, "chunk_overlap": 8, "top_k": 5, "retriever": "bm25"},
        "metrics": {"recall@5": recall, "hit@5": recall, "mrr": recall, "ndcg@5": recall, "n": 14},
    }


def test_changed_variables_single() -> None:
    changed = changed_variables(_record(16, 0.7), _record(32, 0.96))
    assert changed == {"chunk_size": (16, 32)}


def test_metric_deltas_skip_n() -> None:
    rows = metric_deltas(_record(16, 0.70), _record(32, 0.96))
    names = {r[0] for r in rows}
    assert "n" not in names
    recall = next(r for r in rows if r[0] == "recall@5")
    assert round(recall[3], 2) == 0.26


def test_render_names_single_variable() -> None:
    out = render(_record(16, 0.7), _record(32, 0.96))
    assert "single changed variable: chunk_size: 16 -> 32" in out
    assert "recall@5" in out


def test_render_rejects_multi_variable() -> None:
    a = _record(16, 0.7)
    b = _record(32, 0.96)
    b["config"]["top_k"] = 8
    out = render(a, b)
    assert "ERROR" in out


def test_main_roundtrip(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_record(16, 0.7)))
    b.write_text(json.dumps(_record(32, 0.96)))
    assert main([str(a), str(b)]) == 0
    assert "chunk_size" in capsys.readouterr().out


def test_main_usage_error() -> None:
    assert main(["only-one.json"]) == 2
