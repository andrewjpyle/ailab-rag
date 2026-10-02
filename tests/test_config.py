from pathlib import Path
from typing import Any

import pytest

from ailab_rag.config import ConfigError, EvalConfig, load_config


def test_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.lab == "ailab-rag"
    assert cfg.retriever == "bm25"
    assert (cfg.chunk_size, cfg.chunk_overlap, cfg.top_k) == (32, 8, 5)
    assert cfg.primary_metric_key == "recall@5"
    assert cfg.thresholds == {"recall": 0.0, "hit": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_config_defaults_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config(env={}) == EvalConfig()


def test_file_then_env_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        '[eval]\nretriever = "lead"\nchunk_size = 24\ntop_k = 3\n'
        "[thresholds]\nrecall = 0.5\nmrr = 0.3\n"
    )
    cfg = load_config(cfg_file, env={"AILAB_TOP_K": "8", "AILAB_MIN_RECALL": "0.6"})
    assert cfg.retriever == "lead"
    assert cfg.chunk_size == 24
    assert cfg.top_k == 8  # env wins over file
    assert cfg.min_recall == 0.6
    assert cfg.thresholds["mrr"] == 0.3
    assert cfg.primary_metric_key == "recall@8"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_recall": 1.5}, r"\[0, 1\]"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": 4, "chunk_overlap": 4}, "chunk_overlap"),
        ({"top_k": 0}, "top_k"),
        ({"retriever": "dense"}, "retriever"),
        ({"primary_metric": "f2"}, "primary_metric"),
    ],
)
def test_range_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        EvalConfig(**kwargs)


def test_env_int_must_parse(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="integer"):
        load_config(env={"AILAB_TOP_K": "lots"})


def test_env_float_must_parse() -> None:
    with pytest.raises(ConfigError, match="number"):
        load_config(env={"AILAB_TOLERANCE": "wide"})


def test_missing_explicit_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml", env={})


def test_invalid_toml_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text("[eval\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(cfg, env={})


def test_env_points_at_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('[eval]\nretriever = "random"\n')
    assert load_config(env={"AILAB_CONFIG": str(cfg)}).retriever == "random"
