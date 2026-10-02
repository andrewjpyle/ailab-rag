"""Whitespace-token chunker.

``chunk_size`` and ``chunk_overlap`` are measured in WHITESPACE TOKENS: the document
text is split on runs of whitespace, and a chunk is a fixed-length window over that token
list. The window advances by ``stride = chunk_size - chunk_overlap`` tokens. Changing
``chunk_size`` or ``chunk_overlap`` therefore rebuilds the whole chunk set.

Relevance is BY SPAN CONTAINMENT, not by chunk id. A chunk is relevant to a question if
and only if the chunk text contains the question's ``answer_span``, compared
case-insensitively with whitespace normalised. The gold labels (a question plus its
answer span) stay valid across chunk sizes, because relevance is recomputed for each
chunk set.
"""

from __future__ import annotations

from dataclasses import dataclass

from ailab_rag.data import Document, Question


@dataclass(frozen=True, slots=True)
class Chunk:
    """One window of a document, in whitespace tokens.

    ``id`` is ``f"{doc_id}#{index}"``: stable within one chunk set, and it changes when
    the chunking config changes because the window boundaries move.
    """

    doc_id: str
    index: int
    start_tok: int
    end_tok: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace to single spaces. Used for span containment."""
    return " ".join(text.split()).lower()


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split one document into overlapping token windows."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap}")
    tokens = doc.text.split()
    if not tokens:
        return []
    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    index = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(
            Chunk(
                doc_id=doc.id,
                index=index,
                start_tok=start,
                end_tok=end,
                text=" ".join(tokens[start:end]),
            )
        )
        index += 1
        if end == len(tokens):
            break
        start += stride
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks


def relevant_chunk_ids(question: Question, chunks: list[Chunk]) -> set[str]:
    """Return the ids of chunks relevant to ``question`` by span containment.

    A chunk is relevant if its normalised text contains the question's normalised
    ``answer_span``. An unanswerable question (empty span) has no relevant chunk, so its
    retrieval recall is 0.0 by definition, which is correct: there is nothing to find.
    """
    if not question.answer_span:
        return set()
    needle = normalize(question.answer_span)
    return {chunk.id for chunk in chunks if needle in normalize(chunk.text)}
