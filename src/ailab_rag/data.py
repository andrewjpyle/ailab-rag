"""Dataset loading for the RAG lab.

Two JSONL datasets are read here, both validated loudly so a bad fixture fails with a
``file:line`` message instead of silently shrinking the eval set.

* A corpus row is ``{"id": str, "title": str, "text": str}``.
* A golden-question row is
  ``{"id", "question", "doc_id", "answer_span", "answerable": bool, "critical": bool,
  "trap": bool}``.

Each dataset file also carries a sha256 of its raw bytes, recorded in the results so a
number can be traced back to the exact file that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus document."""

    id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Question:
    """One golden question with span-keyed ground truth.

    ``answer_span`` is the exact answer string in the source document. Relevance is
    computed by span containment per chunking config (see :mod:`ailab_rag.chunking`),
    so the gold labels stay valid when the chunk size changes.
    """

    id: str
    question: str
    doc_id: str
    answer_span: str
    answerable: bool
    critical: bool = False
    trap: bool = False


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


def sha256_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path) -> list[tuple[int, dict[str, object]]]:
    """Yield ``(lineno, object)`` pairs, skipping blank lines and validating JSON."""
    rows: list[tuple[int, dict[str, object]]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object")
            rows.append((lineno, row))
    return rows


def _str_field(path: Path, lineno: int, row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{path}:{lineno}: need a non-empty string {key!r}")
    return value


def _bool_field(path: Path, lineno: int, row: dict[str, object], key: str, default: bool) -> bool:
    if key not in row:
        return default
    value = row[key]
    if not isinstance(value, bool):
        raise DatasetError(f"{path}:{lineno}: {key!r} must be a boolean")
    return value


def load_corpus(path: str | Path) -> list[Document]:
    """Load the corpus JSONL. Raises :class:`DatasetError` on any malformed row."""
    path = Path(path)
    docs: list[Document] = []
    seen: set[str] = set()
    for lineno, row in _rows(path):
        doc_id = _str_field(path, lineno, row, "id")
        if doc_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate document id {doc_id!r}")
        seen.add(doc_id)
        docs.append(
            Document(
                id=doc_id,
                title=_str_field(path, lineno, row, "title"),
                text=_str_field(path, lineno, row, "text"),
            )
        )
    if not docs:
        raise DatasetError(f"{path}: corpus is empty")
    return docs


def load_questions(path: str | Path) -> list[Question]:
    """Load the golden-questions JSONL. Raises :class:`DatasetError` on any bad row."""
    path = Path(path)
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, row in _rows(path):
        q_id = _str_field(path, lineno, row, "id")
        if q_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate question id {q_id!r}")
        seen.add(q_id)
        answerable = _bool_field(path, lineno, row, "answerable", default=True)
        span = row.get("answer_span")
        if not isinstance(span, str):
            raise DatasetError(f"{path}:{lineno}: 'answer_span' must be a string")
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str):
            raise DatasetError(f"{path}:{lineno}: 'doc_id' must be a string")
        if answerable and (not span or not doc_id):
            raise DatasetError(
                f"{path}:{lineno}: an answerable question needs a 'doc_id' and an 'answer_span'"
            )
        questions.append(
            Question(
                id=q_id,
                question=_str_field(path, lineno, row, "question"),
                doc_id=doc_id,
                answer_span=span,
                answerable=answerable,
                critical=_bool_field(path, lineno, row, "critical", default=False),
                trap=_bool_field(path, lineno, row, "trap", default=False),
            )
        )
    if not questions:
        raise DatasetError(f"{path}: question set is empty")
    return questions
