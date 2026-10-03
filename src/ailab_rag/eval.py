"""RAG eval runner and regression gate.

Usage::

    ailab-eval                                  # or: python -m ailab_rag
    ailab-eval --retriever bm25 --top-k 8
    ailab-eval --min-recall 0.6 --tolerance 0.05

The gate uses a TOLERANCE BAND, not an exact threshold. Exit codes: 0 = all metrics within
the band, 1 = regression (a metric below ``floor - tolerance``, a baseline that is not
beaten, a saturated corpus, a fabricated answer to an unanswerable question, or an
unsupported citation), 2 = bad config or data (empty corpus, no questions, no chunks).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ailab_core.gate import floor_breach

from ailab_rag.chunking import Chunk, chunk_corpus
from ailab_rag.config import ConfigError, EvalConfig, load_config
from ailab_rag.data import (
    DatasetError,
    Question,
    load_corpus,
    load_questions,
    sha256_file,
)
from ailab_rag.gen_checks import citation_resolvable, classify_refusal, lexical_support
from ailab_rag.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    question_relevance,
    recall_at_k,
    reciprocal_rank,
)
from ailab_rag.providers import (
    CassetteMissError,
    LLMProvider,
    ReplayProvider,
    StubProvider,
)
from ailab_rag.reader import Reader
from ailab_rag.retrieval import Retriever, build_retriever

SCHEMA_VERSION = 1

TABLE_HEADER = (
    "| Date | Commit | Retriever | Dataset | chunk | top_k | recall@k | hit@k | mrr "
    "| ndcg@k | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def current_commit() -> str | None:
    """Commit for the results record: env first (CI, Docker), then git, else None."""
    for var in ("GITHUB_SHA", "AILAB_COMMIT"):
        if sha := os.environ.get(var):
            return sha if sha != "unknown" else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def retrieval_scores(
    retriever: Retriever,
    questions: Sequence[Question],
    chunks: list[Chunk],
    top_k: int,
) -> dict[str, Any]:
    """Retrieval metrics for one retriever over the answerable questions.

    Recall is measured over answerable questions, including those whose answer span is
    split across a chunk boundary (empty relevant set, recall 0.0): a fact the chunking
    makes unretrievable is a real retrieval miss, not an excused one.
    """
    answerable = [q for q in questions if q.answerable]
    per = {"recall": [], "hit": [], "rr": [], "ndcg": [], "recall1": []}  # type: dict[str, list[float]]
    crit: list[float] = []
    trap: list[float] = []
    misses: list[dict[str, Any]] = []
    for q in answerable:
        rel = question_relevance(q, chunks)
        ranked = retriever(q.question, top_k)
        r = recall_at_k(ranked, rel, top_k)
        per["recall"].append(r)
        per["hit"].append(hit_at_k(ranked, rel, top_k))
        per["rr"].append(reciprocal_rank(ranked[:top_k], rel))
        per["ndcg"].append(ndcg_at_k(ranked, rel, top_k))
        per["recall1"].append(recall_at_k(ranked, rel, 1))
        if q.critical:
            crit.append(r)
        if q.trap:
            trap.append(r)
        if r < 1.0:
            misses.append({"id": q.id, "missing": sorted(rel - set(ranked[:top_k]))})
    return {
        "n": len(answerable),
        "recall": mean(per["recall"]),
        "hit": mean(per["hit"]),
        "mrr": mean(per["rr"]),
        "ndcg": mean(per["ndcg"]),
        "recall@1": mean(per["recall1"]),
        "critical_n": len(crit),
        "critical_recall": mean(crit),
        "trap_n": len(trap),
        "trap_recall": mean(trap),
        "misses": misses,
    }


def generation_scores(
    reader: Reader,
    retriever: Retriever,
    questions: Sequence[Question],
    chunks: list[Chunk],
    top_k: int,
) -> dict[str, Any]:
    """Run the Reader over every question and score citations, support and refusals."""
    text_by_id = {c.id: c.text for c in chunks}
    buckets = {"answered": 0, "false_refusal": 0, "refusal_correct": 0, "wrong_refusal": 0}
    resolvable: list[float] = []
    support: list[float] = []
    for q in questions:
        ranked = retriever(q.question, top_k)
        retrieved = [(cid, text_by_id[cid]) for cid in ranked if cid in text_by_id]
        result = reader.answer(q.question, retrieved)
        buckets[classify_refusal(q.answerable, result.refused)] += 1
        if result.refused:
            continue
        resolvable.append(citation_resolvable(result.citations, [cid for cid, _ in retrieved]))
        best = max(
            (lexical_support(result.answer, text_by_id.get(cid, "")) for cid in result.citations),
            default=0.0,
        )
        support.append(best)
    n = len(questions)
    answerable_n = sum(1 for q in questions if q.answerable)
    unanswerable_n = n - answerable_n
    return {
        "answered_n": buckets["answered"],
        "citation_resolvable": mean(resolvable),
        "lexical_support": mean(support),
        "refusal_correct": buckets["refusal_correct"],
        "false_refusal": buckets["false_refusal"],
        "wrong_refusal": buckets["wrong_refusal"],
        "false_refusal_rate": buckets["false_refusal"] / answerable_n if answerable_n else 0.0,
        "wrong_refusal_rate": buckets["wrong_refusal"] / unanswerable_n if unanswerable_n else 0.0,
    }


def _class_counts(questions: Sequence[Question], floor: int) -> dict[str, Any]:
    counts = {
        "answerable": sum(1 for q in questions if q.answerable),
        "unanswerable": sum(1 for q in questions if not q.answerable),
        "trap": sum(1 for q in questions if q.trap),
        "critical": sum(1 for q in questions if q.critical),
    }
    insufficient = [name for name, c in counts.items() if c < floor]
    return {"counts": counts, "insufficient": insufficient, "floor": floor}


def evaluate(config: EvalConfig, reader: Reader | None = None) -> dict[str, Any]:
    """Run one eval and return the results record written to ``eval_results.json``.

    Raises :class:`DatasetError` for empty data or a chunk set of size zero (exit 2).
    """
    docs = load_corpus(config.corpus)
    questions = load_questions(config.questions)
    chunks = chunk_corpus(docs, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise DatasetError(f"{config.corpus}: produced 0 chunks")

    primary = build_retriever(config.retriever, chunks, seed=config.seed)
    bm25 = build_retriever("bm25", chunks, seed=config.seed)
    rnd = build_retriever("random", chunks, seed=config.seed)
    lead = build_retriever("lead", chunks)

    scores = retrieval_scores(primary, questions, chunks, config.top_k)
    bm25_scores = retrieval_scores(bm25, questions, chunks, config.top_k)
    rnd_scores = retrieval_scores(rnd, questions, chunks, config.top_k)
    lead_scores = retrieval_scores(lead, questions, chunks, config.top_k)

    reader = reader or Reader(StubProvider(), support_threshold=config.support_threshold)
    gen = generation_scores(reader, primary, questions, chunks, config.top_k)
    classes = _class_counts(questions, config.insufficient_floor)

    k = config.top_k
    metrics = {
        f"recall@{k}": _round(scores["recall"]),
        f"hit@{k}": _round(scores["hit"]),
        "mrr": _round(scores["mrr"]),
        f"ndcg@{k}": _round(scores["ndcg"]),
        "n": scores["n"],
    }

    baseline_best = max(rnd_scores["recall"] or 0.0, lead_scores["recall"] or 0.0)
    margin = (bm25_scores["recall"] or 0.0) - baseline_best
    recall_k = scores["recall"] or 0.0
    recall_1 = scores["recall@1"] or 0.0
    saturated = recall_k >= 1.0 and recall_1 >= 1.0

    failures = _gate(config, scores, gen, margin, saturated)
    return {
        "schema_version": SCHEMA_VERSION,
        "lab": config.lab,
        "dataset": config.questions.stem,
        "provider": reader.provider.name,
        "model": reader.provider.model,
        "primary_metric": config.primary_metric_key,
        "metrics": metrics,
        "threshold": config.thresholds[config.primary_metric],
        "passed": not failures,
        "commit": current_commit(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        # Diagnostic extras below (not part of the schema_version 1 contract).
        "retriever": config.retriever,
        "corpus_sha256": sha256_file(config.corpus),
        "questions_sha256": sha256_file(config.questions),
        "config": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
            "retriever": config.retriever,
            "support_threshold": config.support_threshold,
            "tolerance": config.tolerance,
        },
        "chunk_count": len(chunks),
        "tolerance": config.tolerance,
        "thresholds": config.thresholds,
        "critical": {"n": scores["critical_n"], "recall": _round(scores["critical_recall"])},
        "trap": {"n": scores["trap_n"], "recall": _round(scores["trap_recall"])},
        "gen_checks": {
            "answered_n": gen["answered_n"],
            "citation_resolvable": _round(gen["citation_resolvable"]),
            "lexical_support": _round(gen["lexical_support"]),
        },
        "refusals": {
            "refusal_correct": gen["refusal_correct"],
            "false_refusal": gen["false_refusal"],
            "wrong_refusal": gen["wrong_refusal"],
            "false_refusal_rate": _round(gen["false_refusal_rate"]),
            "wrong_refusal_rate": _round(gen["wrong_refusal_rate"]),
        },
        "baselines": {
            "bm25_recall": _round(bm25_scores["recall"]),
            "random_recall": _round(rnd_scores["recall"]),
            "lead_recall": _round(lead_scores["recall"]),
            "margin": _round(margin),
            "required_margin": config.baseline_margin,
        },
        "saturation": {
            f"recall@{k}": _round(scores["recall"]),
            "recall@1": _round(scores["recall@1"]),
            "saturated": saturated,
        },
        "classes": classes,
        "misses": scores["misses"],
        "failures": failures,
    }


def _gate(
    config: EvalConfig,
    scores: dict[str, Any],
    gen: dict[str, Any],
    margin: float,
    saturated: bool,
) -> list[str]:
    """Build the list of regression messages; empty means the gate passes."""
    tol = config.tolerance
    failures: list[str] = []
    k = config.top_k
    metric_values = {
        "recall": (f"recall@{k}", scores["recall"]),
        "hit": (f"hit@{k}", scores["hit"]),
        "mrr": ("mrr", scores["mrr"]),
        "ndcg": (f"ndcg@{k}", scores["ndcg"]),
    }
    for base, floor in config.thresholds.items():
        label, value = metric_values[base]
        if msg := floor_breach(label, value, floor, tol):
            failures.append(msg)
    if saturated:
        failures.append(
            f"saturation recall@{k} and recall@1 both 1.0000; corpus cannot detect regressions"
        )
    if margin < config.baseline_margin:
        failures.append(
            f"baseline_margin {margin:.4f} < required {config.baseline_margin:.4f}; "
            "bm25 does not beat the baselines"
        )
    if gen["wrong_refusal_rate"] > config.wrong_refusal_max:
        failures.append(
            f"wrong_refusal {gen['wrong_refusal_rate']:.4f} > max "
            f"{config.wrong_refusal_max:.4f}; an unanswerable question was answered"
        )
    support = gen["lexical_support"]
    answered = gen["answered_n"] > 0
    if (
        answered
        and support is not None
        and (msg := floor_breach("lexical_support", support, config.lexical_support_floor, tol))
    ):
        failures.append(f"{msg}; a citation does not support its answer")
    resolvable = gen["citation_resolvable"]
    if (
        answered
        and resolvable is not None
        and (msg := floor_breach("citation_resolvable", resolvable, config.citation_floor, tol))
    ):
        failures.append(msg)
    return failures


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown_rows(result: dict[str, Any]) -> str:
    """README "Eval results" row, derived only from a results record."""
    commit = (result["commit"] or "unknown")[:7]
    cfg = result["config"]
    k = cfg["top_k"]
    dataset = f"{result['dataset']} (n={result['metrics']['n']})"
    note = "PASS" if result["passed"] else "FAIL"
    return (
        f"| {result['generated_at'][:10]} | {commit} | {result['retriever']} | {dataset} "
        f"| {cfg['chunk_size']}/{cfg['chunk_overlap']} | {k} "
        f"| {_fmt(result['metrics'][f'recall@{k}'])} | {_fmt(result['metrics'][f'hit@{k}'])} "
        f"| {_fmt(result['metrics']['mrr'])} | {_fmt(result['metrics'][f'ndcg@{k}'])} | {note} |"
    )


def _print_summary(result: dict[str, Any]) -> None:
    """Human-readable gate summary printed after the table."""
    classes = result["classes"]
    parts = []
    for name, count in classes["counts"].items():
        token = "INSUFFICIENT" if name in classes["insufficient"] else str(count)
        parts.append(f"{name}={token}")
    print("class n: " + ", ".join(parts))
    sat = result["saturation"]
    k = result["config"]["top_k"]
    print(
        f"saturation guard: recall@{k}={_fmt(sat[f'recall@{k}'])} "
        f"recall@1={_fmt(sat['recall@1'])} -> {'SATURATED' if sat['saturated'] else 'ok'}"
    )
    b = result["baselines"]
    print(
        f"baselines: bm25={_fmt(b['bm25_recall'])} random={_fmt(b['random_recall'])} "
        f"lead={_fmt(b['lead_recall'])} margin={_fmt(b['margin'])} "
        f"(required {b['required_margin']:.2f})"
    )
    r = result["refusals"]
    print(
        f"refusals: correct={r['refusal_correct']} false_refusal={r['false_refusal']} "
        f"wrong_refusal={r['wrong_refusal']}"
    )
    g = result["gen_checks"]
    print(
        f"gen checks: citation_resolvable={_fmt(g['citation_resolvable'])} "
        f"lexical_support={_fmt(g['lexical_support'])} (answered n={g['answered_n']})"
    )
    print(f"critical recall: {_fmt(result['critical']['recall'])} (n={result['critical']['n']})")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ailab-eval", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="TOML config path ($AILAB_CONFIG or eval_config.toml)")
    parser.add_argument("--corpus", type=Path, help="override the corpus JSONL")
    parser.add_argument("--questions", type=Path, help="override the questions JSONL")
    parser.add_argument(
        "--retriever", choices=["bm25", "random", "lead"], help="override retriever"
    )
    parser.add_argument("--chunk-size", type=int, dest="chunk_size", help="override chunk_size")
    parser.add_argument("--chunk-overlap", type=int, dest="chunk_overlap", help="override overlap")
    parser.add_argument("--top-k", type=int, dest="top_k", help="override top_k")
    parser.add_argument("--output", type=Path, help="override the results JSON path")
    parser.add_argument("--min-recall", type=float, dest="min_recall", help="override recall floor")
    parser.add_argument("--tolerance", type=float, help="override the gate tolerance band")
    parser.add_argument(
        "--force-answer",
        action="store_true",
        help="proof-of-gate: never refuse (fabricates answers to unanswerable questions)",
    )
    parser.add_argument(
        "--sabotage-citations",
        action="store_true",
        help="proof-of-gate: cite an unsupporting chunk (lexical-support negative control)",
    )
    parser.add_argument(
        "--provider",
        choices=["stub", "replay"],
        default="stub",
        help="reader backend: offline stub (default) or replay a recorded cassette",
    )
    parser.add_argument(
        "--cassette",
        type=Path,
        default=Path("fixtures/cassettes/nimbus_reader.json"),
        help="cassette path for --provider replay",
    )
    parser.add_argument(
        "--model",
        default="gemma3:27b",
        help="recorded model id to replay (keys are sha256(model, prompt))",
    )
    parser.add_argument(
        "--table-from",
        type=Path,
        metavar="RESULTS_JSON",
        help="print a README table row from an existing results file, without running",
    )
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "corpus",
        "questions",
        "retriever",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "output",
        "min_recall",
        "tolerance",
    )
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.table_from is not None:
        try:
            record = json.loads(args.table_from.read_text(encoding="utf-8"))
            print(f"{TABLE_HEADER}\n{markdown_rows(record)}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ailab-eval: error: cannot read {args.table_from}: {exc!r}", file=sys.stderr)
            return 2
        return 0
    try:
        config = replace(load_config(args.config), **_overrides(args))
        if args.provider == "replay":
            provider: LLMProvider = ReplayProvider(args.cassette, name="ollama", model=args.model)
        else:
            provider = StubProvider()
        reader = Reader(
            provider,
            support_threshold=config.support_threshold,
            force_answer=args.force_answer,
            sabotage_citations=args.sabotage_citations,
        )
        result = evaluate(config, reader=reader)
    except (ConfigError, DatasetError, OSError, CassetteMissError) as exc:
        print(f"ailab-eval: error: {exc}", file=sys.stderr)
        return 2

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    table = f"{TABLE_HEADER}\n{markdown_rows(result)}"
    print(table)
    _print_summary(result)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Eval results\n\n{table}\n\n")

    if not result["passed"]:
        for failure in result["failures"]:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
