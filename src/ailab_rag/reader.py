"""Reader: turn a question plus retrieved chunks into an answer with citations.

Cite-or-refuse: if no retrieved chunk lexically supports the question (token overlap
below ``support_threshold``), the Reader returns a refusal and cites nothing. Otherwise
it asks the provider for an answer and parses the cited chunk ids out of the completion.

The default backend is :class:`ailab_rag.providers.StubProvider`, which is a FORMAT path,
not a real model (see docs/LEARNING.md). The same Reader runs a live provider unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ailab_rag.providers import LLMProvider

_TOKEN = re.compile(r"[a-z0-9]+")

REFUSAL = "I cannot answer this question from the retrieved context."


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def lexical_overlap(question: str, chunk_text: str) -> float:
    """Fraction of the question's content tokens that appear in the chunk."""
    q = _tokens(question)
    return len(q & _tokens(chunk_text)) / len(q) if q else 0.0


@dataclass(frozen=True, slots=True)
class ReaderResult:
    answer: str
    citations: tuple[str, ...]
    refused: bool


PROMPT_TEMPLATE = (
    "Answer the question using only the context. Cite the chunk id you used.\n"
    "Reply in the form:\n"
    "ANSWER: <text>\n"
    "CITE: <chunk_id>\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)


class Reader:
    """Retrieve-then-read answerer with a cite-or-refuse rule."""

    def __init__(
        self,
        provider: LLMProvider,
        support_threshold: float = 0.3,
        *,
        force_answer: bool = False,
        sabotage_citations: bool = False,
    ) -> None:
        self.provider = provider
        self.support_threshold = support_threshold
        self.force_answer = force_answer
        self.sabotage_citations = sabotage_citations

    def build_prompt(self, question: str, retrieved: list[tuple[str, str]]) -> str:
        context = "\n".join(f"[{chunk_id}] {text}" for chunk_id, text in retrieved)
        return PROMPT_TEMPLATE.format(question=question, context=context)

    @staticmethod
    def parse(completion: str) -> tuple[str, list[str]]:
        answer = ""
        citations: list[str] = []
        for line in completion.splitlines():
            if line.startswith("ANSWER:"):
                answer = line[len("ANSWER:") :].strip()
            elif line.startswith("CITE:"):
                citations = [c.strip() for c in line[len("CITE:") :].split(",") if c.strip()]
        return answer, citations

    def answer(self, question: str, retrieved: list[tuple[str, str]]) -> ReaderResult:
        best_support = max((lexical_overlap(question, text) for _, text in retrieved), default=0.0)
        if not self.force_answer and best_support < self.support_threshold:
            return ReaderResult(answer=REFUSAL, citations=(), refused=True)

        completion = self.provider.complete(self.build_prompt(question, retrieved))
        answer, citations = self.parse(completion)
        if self.sabotage_citations and retrieved:
            # Negative control: cite the retrieved chunk that LEAST supports the answer.
            worst_id, _ = min(retrieved, key=lambda item: lexical_overlap(answer, item[1]))
            citations = [worst_id]
        return ReaderResult(answer=answer, citations=tuple(citations), refused=False)
