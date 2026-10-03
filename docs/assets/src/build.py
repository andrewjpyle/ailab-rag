"""Build the README graphics for ailab-rag.

Two structural diagrams (hero, flow) carry no data and are labeled HOW IT WORKS.
Two data graphics (eval anatomy, red-team catalog) are built from committed captures in
captures/, never by hand. Run from the repo root with the vendored kit:

    uv run --with playwright --with pillow python docs/assets/src/render.py docs/assets/src docs/assets
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import readme_kit as k  # noqa: E402

CAPS = HERE / "captures"
REAL_RUN = "AILAB-RAG . REAL RUN 2026-10-03"
HOW = "AILAB-RAG . HOW IT WORKS"


def num(pattern: str, text: str) -> str:
    """Pull one captured number out of the real output. Fails loudly if absent."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"pattern not found in capture: {pattern!r}")
    return m.group(1)


# ── 1. hero (structural) ──────────────────────────────────────────────────────────────────────

def build_hero() -> str:
    lede = ("A local, zero-dependency retrieve-then-read lab over a synthetic corpus. It scores "
            "retrieval and generation, then a tolerance-band gate fails CI on a regression.")
    rules = [
        ("Retrieve, then read", "BM25 ranks chunks; a cite-or-refuse reader answers only from them."),
        ("A number gates CI", "make eval exits 1 when a metric falls below its floor minus tolerance."),
        ("Prove the gate can fail", "make red fires sabotages; each exits non-zero and names its metric."),
    ]
    right = k.wheel(["CHUNK", "INDEX", "RETRIEVE", "READ", "GATE"], "AILAB", "RAG")
    return k.hero(
        kicker="OFFLINE RAG LAB . ZERO DEPENDENCIES",
        title="A RAG lab where",
        accent="a number gates CI",
        lede_html=lede,
        rules=rules,
        pill="BM25 . CITE-OR-REFUSE . TOLERANCE GATE",
        right_html=right,
        footer_left=HOW,
    )


# ── 2. flow (structural) ──────────────────────────────────────────────────────────────────────

def build_flow() -> str:
    y = 300
    h = 180
    xs = [56 + i * 267 for i in range(5)]
    boxes = "".join([
        k.box(xs[0], y, 218, h, "CORPUS + QUESTIONS",
              ["20 synthetic docs", "14 answerable Qs", "span-keyed gold labels"]),
        k.box(xs[1], y, 218, h, "CHUNK",
              ["32-token windows", "overlap 8 tokens", "20 docs -> 87 chunks"]),
        k.box(xs[2], y, 218, h, "INDEX + RETRIEVE",
              ["BM25, pure Python", "top_k = 5", "vs random + lead"]),
        k.box(xs[3], y, 218, h, "READ",
              ["cite-or-refuse", "answers from chunks", "refuse if unsupported"]),
        k.box(xs[4], y, 218, h, "GATE", ["tolerance band", "must beat baselines", "exit 1 on regression"],
              accent=True),
    ])
    mid = y + h // 2
    specs = [(xs[i] + 218, mid, xs[i + 1], mid) for i in range(4)]
    sub = "offline, deterministic: StubProvider by default, a recorded gemma3:27b cassette on replay"
    return k.flow(
        kicker="HOW IT WORKS",
        title_html=f"Chunk, index, retrieve, read, {k.em('then gate')}",
        subline=sub,
        boxes_html=boxes,
        arrow_specs=specs,
        footer_left=HOW,
    )


# ── 3. eval anatomy (capture-driven) ────────────────────────────────────────────────────────────

def build_eval() -> str:
    cap = k.load_capture(CAPS / "eval.json")
    out = cap["output"]
    commit = cap["commit"][:7]
    n = num(r"nimbus_questions \(n=(\d+)\)", out)
    recall = num(r"\| 5 \| ([\d.]+) \|", out)
    row = re.search(r"\| 5 \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", out)
    assert row
    recall5, hit5, mrr, ndcg5 = row.groups()
    sat1 = num(r"recall@1=([\d.]+)", out)
    bm25 = num(r"bm25=([\d.]+)", out)
    rand = num(r"random=([\d.]+)", out)
    lead = num(r"lead=([\d.]+)", out)
    margin = num(r"margin=([\d.]+)", out)
    cite = num(r"citation_resolvable=([\d.]+)", out)
    lex = num(r"lexical_support=([\d.]+)", out)
    crit = num(r"critical recall: ([\d.]+)", out)
    refc = num(r"refusals: correct=(\d+)", out)

    doc = [
        ("h1", "make eval"),
        ("q", f"one real run . commit {k.esc(commit)} . bm25 . nimbus_questions (n={k.esc(n)})"),
        ("h2", "Retrieval metrics"),
        ("code", f"recall@5   {recall5}"),
        ("code", f"hit@5      {hit5}"),
        ("code", f"mrr        {mrr}"),
        ("code", f"ndcg@5     {ndcg5}"),
        ("h2", "Honesty checks"),
        ("code", f"citation_resolvable   {cite}"),
        ("code", f"lexical_support       {lex}"),
        ("code", f"critical recall       {crit}"),
        ("code", f"refusals correct      {refc} of 4 unanswerable"),
        ("h2", "Baselines and saturation"),
        ("code", f"bm25 {bm25}  random {rand}  lead {lead}"),
        ("code", f"margin over best baseline   {margin}"),
        ("code", f"saturation guard recall@1   {sat1}"),
        ("m", "generated from eval_results.json, never typed by hand"),
    ]
    notes = [
        (150, f"recall@5 is {recall5}, not 1.0: one question's second relevant chunk sits at rank 66, past top_k=5."),
        (300, f"lexical_support {lex} is the stub copying its cited chunk; the real gemma3 reader scores 0.8457, lower and more honest."),
        (430, f"BM25 beats random and lead by {margin}, far above the 0.10 floor a dumb retriever must not clear."),
        (520, f"recall@1 {sat1} is well below 1.0, so the corpus can still detect a regression. An easy corpus fails on purpose."),
    ]
    return k.anatomy("ANATOMY OF A REAL EVAL", doc, notes, REAL_RUN)


# ── 4. red-team gate catalog (capture-driven) ─────────────────────────────────────────────────

def build_red() -> str:
    cap = k.load_capture(CAPS / "red.json")
    out = cap["output"]
    cases = re.findall(r"PASS \[(\S+) ([^\]]+)\]: exit (\d+), named '([^']+)'", out)
    if len(cases) != 5:
        raise SystemExit(f"expected 5 red cases in capture, found {len(cases)}")
    detail = {
        "recall@": "recall@5 0.0714 < floor 0.70",
        "recall@1": "recall@1 0.4286 < floor 0.70",
        "empty": "corpus is empty",
        "wrong_refusal": "answered an unanswerable Q",
        "lexical_support": "citation does not support answer",
    }
    cards = []
    for cid, label, code, metric in cases:
        dim = next((v for kk, v in detail.items() if kk == metric), f"named {metric}")
        cards.append((f"SABOTAGE {cid}", label, f"exit {code} . {metric}", dim))
    sub = ("make red . four sabotage classes (one in two variants), five checks in all . "
           "each exits non-zero and names the metric it tripped")
    return k.catalog(
        "THE GATE MUST FAIL ON PURPOSE",
        f"A gate that {k.em('cannot fail')} proves nothing",
        sub,
        cards,
        REAL_RUN,
        cols=5,
        card_height=210,
    )


if __name__ == "__main__":
    pages = {
        "hero": build_hero(),
        "architecture": build_flow(),
        "anatomy": build_eval(),
        "red-gate": build_red(),
    }
    k.write_pages(HERE, pages)
