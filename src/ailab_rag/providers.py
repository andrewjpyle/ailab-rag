"""LLM provider seam for the Reader. Stdlib only: no SDKs, no third-party HTTP.

* :class:`StubProvider` is a deterministic offline Reader backend. It picks the context
  chunk with the most token overlap with the question and echoes a short span from it, so
  the prompt -> completion -> parse path runs with no network. Stub answers are a FORMAT
  path, not a real model (see docs/LEARNING.md).
* :class:`ReplayProvider` serves a committed JSON cassette keyed by ``sha256(model,
  prompt)``. A miss RAISES, so a stale cassette cannot silently pass.
* :class:`OllamaProvider` POSTs to a local Ollama server with ``urllib`` from stdlib, used
  only in record mode to fill a cassette.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")
_CONTEXT_LINE = re.compile(r"^\[([^\]]+)\]\s+(.*)$")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@runtime_checkable
class LLMProvider(Protocol):
    """The minimum a text-completion backend must offer.

    A real implementation wraps a vendor SDK or a local model server and reads its
    credentials from the environment. Keeping the protocol this narrow is what makes
    providers swappable and the eval reproducible.
    """

    name: str  # provider id recorded in results, e.g. "stub"
    model: str

    def complete(self, prompt: str) -> str: ...


def _extract_question(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Question:"):
            return line[len("Question:") :].strip()
    return ""


def _extract_context(prompt: str) -> list[tuple[str, str]]:
    """Parse ``[chunk_id] text`` lines from the Context block of a Reader prompt."""
    out: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        match = _CONTEXT_LINE.match(line.strip())
        if match:
            out.append((match.group(1), match.group(2)))
    return out


class StubProvider:
    """Deterministic offline Reader backend. No network, no key."""

    name = "stub"
    model = "stub-reader-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        question = _extract_question(prompt)
        context = _extract_context(prompt)
        if not context:
            return "ANSWER: I cannot answer from the provided context.\nCITE:"
        q_tokens = _tokens(question)
        # Pick the chunk that shares the most tokens with the question (stable tie-break).
        best_id, best_text = max(
            context,
            key=lambda item: (len(q_tokens & _tokens(item[1])), -context.index(item)),
        )
        span = " ".join(best_text.split()[:25])
        return f"ANSWER: {span}\nCITE: {best_id}"


def cassette_key(model: str, prompt: str) -> str:
    """Stable key for a cassette entry: ``sha256(model, prompt)``."""
    raw = json.dumps([model, prompt], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CassetteMissError(RuntimeError):
    """Raised when a ReplayProvider is asked for a prompt its cassette does not hold."""


class ReplayProvider:
    """Replay a committed JSON cassette; a miss raises so a stale cassette cannot pass.

    The cassette is a JSON object mapping ``sha256(model, prompt)`` to
    ``{"model", "response"}``. ``name``/``model`` mirror the recorded backend so the
    results record names what was replayed.
    """

    def __init__(self, path: str | Path, name: str = "replay", model: str = "replay") -> None:
        self.path = Path(path)
        self.name = name
        self.model = model
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries: dict[str, dict[str, str]] = raw.get("entries", raw)

    def complete(self, prompt: str) -> str:
        key = cassette_key(self.model, prompt)
        hit = self._entries.get(key)
        if hit is None:
            raise CassetteMissError(
                f"cassette miss for model {self.model!r} ({key[:12]}); re-record the cassette"
            )
        return hit["response"]


class OllamaProvider:
    """Local models via Ollama ``/api/generate``. Used only in record mode. Stdlib HTTP."""

    name = "ollama"

    def __init__(self, model: str, host: str | None = None, timeout: float = 300.0) -> None:
        self.model = model
        base = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not base.startswith("http"):
            base = "http://" + base
        self.url = base.rstrip("/") + "/api/generate"
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
            raise RuntimeError(f"ollama: {exc}") from exc
        response = data.get("response", "")
        return str(response)
