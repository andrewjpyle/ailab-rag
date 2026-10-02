#!/usr/bin/env python3
"""Record a REAL Reader cassette from a live Ollama model.

It rebuilds the exact pipeline the eval uses at the canonical config (bm25, the configured
chunk_size/overlap and top_k) and asks the model each golden question with the SAME
``Reader.build_prompt`` text. It writes ``fixtures/cassettes/nimbus_reader.json`` keyed by
``sha256(model, prompt)`` so ``ReplayProvider`` can look the answers up later with no network.

The model id comes from ``AILAB_RECORD_MODEL`` (default ``gemma3:27b``). The server URL comes
from ``OLLAMA_HOST`` and is never written to disk, because the cassette stores only the
prompt hash and the response.

Usage::

    OLLAMA_HOST=http://<host>:11434 make record
    OLLAMA_HOST=http://<host>:11434 AILAB_RECORD_MODEL=gemma3:27b \
        uv run python scripts/record_cassette.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ailab_rag.chunking import chunk_corpus  # noqa: E402
from ailab_rag.config import load_config  # noqa: E402
from ailab_rag.data import load_corpus, load_questions  # noqa: E402
from ailab_rag.providers import OllamaProvider, cassette_key  # noqa: E402
from ailab_rag.reader import Reader  # noqa: E402
from ailab_rag.retrieval import BM25  # noqa: E402

CASSETTE = REPO / "fixtures" / "cassettes" / "nimbus_reader.json"


def _get_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out: dict[str, Any] = json.loads(resp.read())
    return out


def main() -> int:
    if not os.environ.get("OLLAMA_HOST"):
        print("record_cassette: set OLLAMA_HOST (never hardcoded here)", file=sys.stderr)
        return 2
    model = os.environ.get("AILAB_RECORD_MODEL", "gemma3:27b")
    config = load_config()
    docs = load_corpus(config.corpus)
    questions = load_questions(config.questions)
    chunks = chunk_corpus(docs, config.chunk_size, config.chunk_overlap)
    text_by_id = {c.id: c.text for c in chunks}
    bm25 = BM25(chunks)
    provider = OllamaProvider(model)
    reader = Reader(provider)

    # Provenance metadata (no host): Ollama version and the model digest, so a re-record is
    # identifiable and reproducible. Failures here are non-fatal.
    version = ""
    digest = ""
    try:
        version = str(_get_json(f"{provider.base}/api/version").get("version", ""))
        tags = _get_json(f"{provider.base}/api/tags").get("models", [])
        digest = next((str(m.get("digest", "")) for m in tags if m.get("name") == model), "")
    except Exception as exc:
        print(f"record_cassette: metadata lookup failed ({exc}); continuing", file=sys.stderr)

    entries: dict[str, dict[str, Any]] = {}
    for i, q in enumerate(questions, start=1):
        ranked = bm25(q.question, config.top_k)
        retrieved = [(cid, text_by_id[cid]) for cid in ranked]
        prompt = reader.build_prompt(q.question, retrieved)
        print(f"[{i}/{len(questions)}] {q.id}: calling {model} ...", flush=True)
        raw = provider.generate(prompt)
        response = str(raw.get("response", ""))
        if not response.strip():
            print(f"record_cassette: empty response for {q.id}; aborting", file=sys.stderr)
            return 1
        entries[cassette_key(model, prompt)] = {
            "model": model,
            "response": response,
            # Vendor's real token counts (not a whitespace estimate).
            "prompt_eval_count": int(raw.get("prompt_eval_count", 0) or 0),
            "eval_count": int(raw.get("eval_count", 0) or 0),
        }

    payload = {
        "model": model,
        "provider": "ollama",
        "model_digest": digest,
        "ollama_version": version,
        "temperature": 0,
        "seed": OllamaProvider.DEFAULT_SEED,
        "recorded_on": datetime.now(UTC).date().isoformat(),
        "note": (
            "Real recording from a local Ollama server (host from OLLAMA_HOST, never stored). "
            "Keys are sha256(model, prompt) built from the canonical pipeline (bm25, the "
            "configured chunk_size/overlap and top_k). Pinned temperature 0 and a fixed seed "
            "make a re-record deterministic."
        ),
        "entries": entries,
    }
    CASSETTE.parent.mkdir(parents=True, exist_ok=True)
    CASSETTE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"record_cassette: wrote {len(entries)} entries for model {model} to {CASSETTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
