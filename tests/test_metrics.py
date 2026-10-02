from ailab_rag.chunking import chunk_corpus
from ailab_rag.data import Document, Question
from ailab_rag.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    question_relevance,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_hit_rr_ndcg_known_values() -> None:
    ranked = ["a", "b", "c"]
    rel = {"b"}
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert hit_at_k(ranked, rel, 3) == 1.0
    assert hit_at_k(ranked, rel, 1) == 0.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert ndcg_at_k(ranked, {"a"}, 3) == 1.0  # top rank is ideal


def test_empty_relevant_set_scores_zero() -> None:
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert hit_at_k(["a"], set(), 3) == 0.0
    assert ndcg_at_k(["a"], set(), 3) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0


def test_mean_handles_empty() -> None:
    assert mean([]) is None
    assert mean([1.0, 0.0]) == 0.5


def test_question_relevance_matches_span_containment() -> None:
    doc = Document(id="d", title="T", text="The AES two fifty six cipher protects data at rest")
    chunks = chunk_corpus([doc], chunk_size=4, chunk_overlap=1)
    q = Question(id="q", question="?", doc_id="d", answer_span="fifty six cipher", answerable=True)
    rel = question_relevance(q, chunks)
    assert rel
    assert all(cid.startswith("d#") for cid in rel)
