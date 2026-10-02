import pytest

from ailab_rag.chunking import (
    Chunk,
    chunk_corpus,
    chunk_document,
    normalize,
    relevant_chunk_ids,
)
from ailab_rag.data import Document, Question


def _doc(text: str, doc_id: str = "d") -> Document:
    return Document(id=doc_id, title="T", text=text)


def test_chunk_id_format() -> None:
    chunk = Chunk(doc_id="d", index=2, start_tok=0, end_tok=3, text="a b c")
    assert chunk.id == "d#2"


def test_windows_and_overlap() -> None:
    doc = _doc(" ".join(str(i) for i in range(10)))
    chunks = chunk_document(doc, chunk_size=4, chunk_overlap=1)
    # stride = 3: windows [0:4], [3:7], [6:10]; the last ends exactly at len, so no tail.
    assert [(c.start_tok, c.end_tok) for c in chunks] == [(0, 4), (3, 7), (6, 10)]
    assert chunks[0].text == "0 1 2 3"
    assert [c.index for c in chunks] == [0, 1, 2]


def test_no_overlap_tiles_exactly() -> None:
    doc = _doc(" ".join(str(i) for i in range(6)))
    chunks = chunk_document(doc, chunk_size=3, chunk_overlap=0)
    assert [c.text for c in chunks] == ["0 1 2", "3 4 5"]


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document(_doc("   "), 4, 1) == []


def test_changing_chunk_size_rebuilds_the_set() -> None:
    doc = _doc(" ".join(str(i) for i in range(20)))
    small = {c.id for c in chunk_document(doc, 4, 1)}
    large = {c.id for c in chunk_document(doc, 8, 1)}
    assert small != large


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (3, 3), (3, 4), (3, -1)],
)
def test_bad_params_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"chunk_size|chunk_overlap"):
        chunk_document(_doc("a b c d"), size, overlap)


def test_relevance_is_by_span_containment_case_insensitive() -> None:
    chunks = chunk_corpus([_doc("The Relay uploads every ninety seconds by default")], 4, 1)
    q = Question(id="q", question="?", doc_id="d", answer_span="NINETY Seconds", answerable=True)
    rel = relevant_chunk_ids(q, chunks)
    assert rel
    assert all("#" in cid for cid in rel)


def test_span_split_across_boundary_has_no_relevant_chunk() -> None:
    # "b c d e f" spans beyond any size-2 window, so no chunk contains it.
    chunks = chunk_corpus([_doc("a b c d e f g")], chunk_size=2, chunk_overlap=0)
    q = Question(id="q", question="?", doc_id="d", answer_span="b c d e f", answerable=True)
    assert relevant_chunk_ids(q, chunks) == set()
    # A larger window contains it.
    big = chunk_corpus([_doc("a b c d e f g")], chunk_size=8, chunk_overlap=0)
    assert relevant_chunk_ids(q, big)


def test_unanswerable_question_has_empty_relevance() -> None:
    chunks = chunk_corpus([_doc("a b c d")], 4, 0)
    q = Question(id="u", question="?", doc_id="", answer_span="", answerable=False)
    assert relevant_chunk_ids(q, chunks) == set()


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  Hello   WORLD\n") == "hello world"
