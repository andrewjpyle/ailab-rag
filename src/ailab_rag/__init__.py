"""ailab_rag: a small, eval-gated, local-only retrieve-then-read RAG lab."""

from ailab_rag.chunking import Chunk, chunk_corpus, relevant_chunk_ids
from ailab_rag.data import Document, Question, load_corpus, load_questions
from ailab_rag.gen_checks import citation_resolvable, classify_refusal, lexical_support
from ailab_rag.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from ailab_rag.providers import LLMProvider, OllamaProvider, ReplayProvider, StubProvider
from ailab_rag.reader import Reader, ReaderResult
from ailab_rag.retrieval import BM25, LeadRetriever, RandomRetriever, Retriever, build_retriever

__all__ = [
    "BM25",
    "Chunk",
    "Document",
    "LLMProvider",
    "LeadRetriever",
    "OllamaProvider",
    "Question",
    "RandomRetriever",
    "Reader",
    "ReaderResult",
    "ReplayProvider",
    "Retriever",
    "StubProvider",
    "build_retriever",
    "chunk_corpus",
    "citation_resolvable",
    "classify_refusal",
    "hit_at_k",
    "lexical_support",
    "load_corpus",
    "load_questions",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevant_chunk_ids",
]

__version__ = "0.1.0"
