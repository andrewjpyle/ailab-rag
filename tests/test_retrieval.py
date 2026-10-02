import pytest

from ailab_rag.chunking import Chunk, chunk_corpus
from ailab_rag.data import Document
from ailab_rag.retrieval import (
    BM25,
    LeadRetriever,
    RandomRetriever,
    Retriever,
    build_retriever,
    tokenize,
)


def _chunks() -> list[Chunk]:
    docs = [
        Document(id="d1", title="A", text="alpha beta gamma delta epsilon zeta eta theta"),
        Document(id="d2", title="B", text="refund billing invoice charge payment receipt plan"),
        Document(id="d3", title="C", text="crash error bug broken freeze stuck glitch fail"),
    ]
    return chunk_corpus(docs, chunk_size=4, chunk_overlap=0)


def test_tokenize() -> None:
    assert tokenize("Refund, MY Card!") == ["refund", "my", "card"]


def test_bm25_ranks_the_matching_chunk_first() -> None:
    chunks = _chunks()
    bm = BM25(chunks)
    ranked = bm("refund invoice", 3)
    assert ranked
    assert ranked[0].startswith("d2#")


def test_bm25_scored_returns_positive_scores_only() -> None:
    chunks = _chunks()
    scored = BM25(chunks).scored("refund invoice", 5)
    assert all(s > 0 for _, s in scored)
    assert [cid for cid, _ in scored] == BM25(chunks)("refund invoice", 5)


def test_bm25_no_match_returns_empty() -> None:
    assert BM25(_chunks())("zzz qqq nomatch", 3) == []


def test_random_retriever_is_seeded_and_deterministic() -> None:
    chunks = _chunks()
    a = RandomRetriever(chunks, seed=0)("refund", 3)
    b = RandomRetriever(chunks, seed=0)("refund", 3)
    assert a == b
    assert len(a) == 3


def test_lead_retriever_returns_only_lead_chunks() -> None:
    chunks = _chunks()
    lead = LeadRetriever(chunks)("anything", 5)
    assert all(cid.endswith("#0") for cid in lead)


def test_build_retriever_factory() -> None:
    chunks = _chunks()
    for name in ("bm25", "random", "lead"):
        r = build_retriever(name, chunks)
        assert isinstance(r, Retriever)
        assert r.name == name
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("dense", chunks)


def test_bm25_finds_the_keyword_chunk_the_lead_baseline_cannot() -> None:
    chunks = _chunks()
    bm = BM25(chunks)("crash bug", 1)
    lead = LeadRetriever(chunks)("crash bug", 3)
    assert bm == ["d3#0"]
    # The lead baseline returns lead chunks in document order, not by relevance.
    assert lead[0] == "d1#0"
