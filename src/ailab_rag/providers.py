"""LLM provider seam for the Reader.

The discipline-agnostic pieces (the ``LLMProvider`` protocol, ``cassette_key``,
``CassetteMissError``, ``ReplayProvider`` and ``OllamaProvider``) now live in ``ailab_core`` and
are re-exported here so existing imports keep working. Only :class:`StubProvider` is rag-specific:
its output is a Reader answer (a context span), so it cannot be shared across labs.
"""

from __future__ import annotations

import re

from ailab_core.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    SupportsTokenCounts,
    cassette_key,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_CONTEXT_LINE = re.compile(r"^\[([^\]]+)\]\s+(.*)$")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


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


__all__ = [
    "CassetteMissError",
    "LLMProvider",
    "OllamaProvider",
    "ReplayProvider",
    "StubProvider",
    "SupportsTokenCounts",
    "cassette_key",
]
