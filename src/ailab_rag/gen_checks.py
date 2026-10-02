"""Deterministic generation checks for the Reader. No model, no network.

* :func:`citation_resolvable` asks whether every cited id is in the retrieved set. The
  stub metric is called ``citation_resolvable``, NOT ``faithfulness``: a resolvable
  citation only proves the id exists, not that the chunk supports the claim.
* :func:`lexical_support` scores claim n-gram overlap between the answer and the cited
  chunk text. A citation to an unsupporting chunk scores near zero (the negative control
  in ``tests/test_gen_checks.py``).
* :func:`classify_refusal` buckets each answered question so the runner can report
  ``refusal_correct``, ``false_refusal`` (refused an answerable one) and ``wrong_refusal``
  (answered an unanswerable one).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {(t,) for t in tokens}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def citation_resolvable(cited_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """Fraction of cited ids that are in the retrieved set. 1.0 means all resolve.

    An answer with no citation scores 0.0: an uncited claim is not resolvable.
    """
    if not cited_ids:
        return 0.0
    retrieved = set(retrieved_ids)
    return sum(c in retrieved for c in cited_ids) / len(cited_ids)


def lexical_support(answer: str, cited_text: str, n: int = 2) -> float:
    """Claim n-gram overlap: fraction of answer n-grams present in the cited chunk.

    Uses bigrams by default, falling back to unigrams for a one-word answer. A high score
    means the cited chunk literally contains the answer's phrasing; a citation to an
    unsupporting chunk scores near zero.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    # Use the same n for both sides, falling back to unigrams for a short answer, so the
    # answer and chunk n-grams are comparable.
    effective_n = min(n, len(answer_tokens))
    answer_grams = _ngrams(answer_tokens, effective_n)
    cited_grams = _ngrams(_tokens(cited_text), effective_n)
    return len(answer_grams & cited_grams) / len(answer_grams)


def classify_refusal(answerable: bool, refused: bool) -> str:
    """Bucket one question's outcome for the refusal report."""
    if answerable:
        return "false_refusal" if refused else "answered"
    return "refusal_correct" if refused else "wrong_refusal"
