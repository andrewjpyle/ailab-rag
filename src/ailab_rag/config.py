"""RAG eval configuration: a TOML file, overridable by environment variables (12-factor).

Layering is defaults <- TOML file <- environment. The CLI adds a final override layer in
:mod:`ailab_rag.eval`. Ranges are validated up front so a bad config fails loudly.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "eval_config.toml"

RETRIEVERS = ("bm25", "random", "lead")
METRIC_BASES = ("recall", "hit", "mrr", "ndcg")

#: environment variable -> EvalConfig attribute
ENV_OVERRIDES: dict[str, str] = {
    "AILAB_LAB": "lab",
    "AILAB_CORPUS": "corpus",
    "AILAB_QUESTIONS": "questions",
    "AILAB_RETRIEVER": "retriever",
    "AILAB_CHUNK_SIZE": "chunk_size",
    "AILAB_CHUNK_OVERLAP": "chunk_overlap",
    "AILAB_TOP_K": "top_k",
    "AILAB_OUTPUT": "output",
    "AILAB_MIN_RECALL": "min_recall",
    "AILAB_MIN_HIT": "min_hit",
    "AILAB_MIN_MRR": "min_mrr",
    "AILAB_MIN_NDCG": "min_ndcg",
    "AILAB_TOLERANCE": "tolerance",
    "AILAB_PRIMARY_METRIC": "primary_metric",
    "AILAB_SUPPORT_THRESHOLD": "support_threshold",
}

_INT_ATTRS = {"chunk_size", "chunk_overlap", "top_k"}
_FLOAT_ATTRS = {
    "min_recall",
    "min_hit",
    "min_mrr",
    "min_ndcg",
    "tolerance",
    "support_threshold",
}
_PATH_ATTRS = {"corpus", "questions", "output"}


class ConfigError(ValueError):
    """Raised for a missing or invalid configuration value."""


@dataclass(frozen=True, slots=True)
class EvalConfig:
    lab: str = "ailab-rag"
    corpus: Path = Path("fixtures/nimbus_corpus.jsonl")
    questions: Path = Path("fixtures/nimbus_questions.jsonl")
    retriever: str = "bm25"
    chunk_size: int = 32
    chunk_overlap: int = 8
    top_k: int = 5
    output: Path = Path("eval_results.json")
    min_recall: float = 0.0
    min_hit: float = 0.0
    min_mrr: float = 0.0
    min_ndcg: float = 0.0
    tolerance: float = 0.05
    primary_metric: str = "recall"
    support_threshold: float = 0.40
    lexical_support_floor: float = 0.3
    citation_floor: float = 1.0
    baseline_margin: float = 0.10
    insufficient_floor: int = 3
    wrong_refusal_max: float = 0.0
    seed: int = 0
    thresholds: dict[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in _FLOAT_ATTRS | {"lexical_support_floor", "citation_floor", "baseline_margin"}:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be within [0, 1], got {value}")
        if self.chunk_size <= 0:
            raise ConfigError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError(f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap}")
        if self.top_k < 1:
            raise ConfigError(f"top_k must be at least 1, got {self.top_k}")
        if self.retriever not in RETRIEVERS:
            raise ConfigError(f"retriever must be one of {RETRIEVERS}, not {self.retriever!r}")
        if self.primary_metric not in METRIC_BASES:
            raise ConfigError(
                f"primary_metric must be one of {METRIC_BASES}, not {self.primary_metric!r}"
            )
        object.__setattr__(
            self,
            "thresholds",
            {
                "recall": self.min_recall,
                "hit": self.min_hit,
                "mrr": self.min_mrr,
                "ndcg": self.min_ndcg,
            },
        )

    @property
    def primary_metric_key(self) -> str:
        """The primary metric as it appears in the metrics dict, e.g. ``recall@5``."""
        return f"{self.primary_metric}@{self.top_k}"


def _int(name: str, raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _from_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    section = data.get("eval", {})
    thresholds = data.get("thresholds", {})
    values: dict[str, Any] = {}
    str_keys = ("lab", "retriever", "primary_metric")
    for key in str_keys:
        if key in section:
            values[key] = str(section[key])
    for key in ("corpus", "questions", "output"):
        if key in section:
            values[key] = Path(section[key])
    for key in ("chunk_size", "chunk_overlap", "top_k"):
        if key in section:
            values[key] = _int(f"eval.{key}", section[key])
    for key in ("tolerance", "support_threshold"):
        if key in section:
            values[key] = _float(f"eval.{key}", section[key])
    threshold_map = {
        "recall": "min_recall",
        "hit": "min_hit",
        "mrr": "min_mrr",
        "ndcg": "min_ndcg",
    }
    for toml_key, attr in threshold_map.items():
        if toml_key in thresholds:
            values[attr] = _float(f"thresholds.{toml_key}", thresholds[toml_key])
    return values


def _apply_env(config: EvalConfig, env: Mapping[str, str]) -> EvalConfig:
    updates: dict[str, Any] = {}
    for var, attr in ENV_OVERRIDES.items():
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        if attr in _PATH_ATTRS:
            updates[attr] = Path(raw)
        elif attr in _INT_ATTRS:
            updates[attr] = _int(var, raw)
        elif attr in _FLOAT_ATTRS:
            updates[attr] = _float(var, raw)
        else:
            updates[attr] = raw
    return replace(config, **updates) if updates else config


def load_config(
    path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> EvalConfig:
    """Build the config: defaults <- TOML file <- environment variables.

    ``path`` defaults to ``$AILAB_CONFIG`` or ``eval_config.toml``. An explicitly
    requested file that does not exist is an error; the implicit default is optional.
    """
    env = os.environ if env is None else env
    explicit = path is not None or bool(env.get("AILAB_CONFIG"))
    config_path = Path(path or env.get("AILAB_CONFIG") or DEFAULT_CONFIG_PATH)

    values: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("rb") as handle:
            try:
                values = _from_mapping(tomllib.load(handle))
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(f"{config_path}: invalid TOML ({exc})") from exc
    elif explicit:
        raise ConfigError(f"config file not found: {config_path}")

    return _apply_env(EvalConfig(**values), env)
