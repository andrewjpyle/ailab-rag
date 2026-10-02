from pathlib import Path

import pytest

from ailab_rag.data import (
    DatasetError,
    load_corpus,
    load_questions,
    sha256_file,
)
from tests.conftest import CORPUS, QUESTIONS


def test_bundled_corpus_loads() -> None:
    docs = load_corpus(CORPUS)
    assert len(docs) == 20
    assert len({d.id for d in docs}) == 20
    assert all(d.title and d.text for d in docs)


def test_bundled_questions_load() -> None:
    questions = load_questions(QUESTIONS)
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    assert len(answerable) >= 8
    assert len(unanswerable) >= 3
    assert sum(q.trap for q in questions) >= 2
    assert sum(q.critical for q in questions) >= 1
    # Every unanswerable question has no doc_id or span.
    assert all(not q.doc_id and not q.answer_span for q in unanswerable)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text("hello")
    assert sha256_file(f) == sha256_file(f)
    assert len(sha256_file(f)) == 64


def test_corpus_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text('{"id":"a","title":"A","text":"x"}\n\n{"id":"b","title":"B","text":"y"}\n')
    assert [d.id for d in load_corpus(p)] == ["a", "b"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", "invalid JSON"),
        ("[1,2]\n", "JSON object"),
        ('{"id":"a","title":"A"}\n', "text"),
        ('{"id":"a","title":"","text":"x"}\n', "title"),
        ("\n\n", "empty"),
    ],
)
def test_corpus_rejects_malformed(tmp_path: Path, content: str, message: str) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(content)
    with pytest.raises(DatasetError, match=message):
        load_corpus(p)


def test_corpus_rejects_duplicate_id(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text('{"id":"a","title":"A","text":"x"}\n{"id":"a","title":"B","text":"y"}\n')
    with pytest.raises(DatasetError, match="duplicate document id"):
        load_corpus(p)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id":"q","question":"x","doc_id":"d"}\n', "answer_span"),
        ('{"id":"q","question":"x","doc_id":"d","answer_span":5}\n', "answer_span"),
        ('{"id":"q","question":"x","doc_id":"","answer_span":"s"}\n', "doc_id"),
        ('{"id":"q","question":"x","doc_id":"d","answer_span":"s","critical":"yes"}\n', "critical"),
        ("\n\n", "empty"),
    ],
)
def test_questions_reject_malformed(tmp_path: Path, content: str, message: str) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(content)
    with pytest.raises(DatasetError, match=message):
        load_questions(p)


def test_questions_reject_duplicate_id(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text(
        '{"id":"q","question":"a","doc_id":"d","answer_span":"s"}\n'
        '{"id":"q","question":"b","doc_id":"d","answer_span":"s"}\n'
    )
    with pytest.raises(DatasetError, match="duplicate question id"):
        load_questions(p)


def test_unanswerable_question_allows_empty_span(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text('{"id":"u","question":"?","doc_id":"","answer_span":"","answerable":false}\n')
    q = load_questions(p)[0]
    assert q.answerable is False
    assert q.doc_id == ""
