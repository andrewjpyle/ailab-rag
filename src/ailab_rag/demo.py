"""Demo: run the retrieve-then-read pipeline on a few questions, then run the eval.

It PRINTS every stage: the chunk count, the top-k retrieved chunk ids with scores, the
selected context, and the answer with citations (or the refusal). It also prints one
worked example of a relevant chunk EXCLUDED by the top-k cutoff, with its rank and k, so
the cost of the cutoff is visible rather than hidden.
"""

from __future__ import annotations

from collections.abc import Sequence

from ailab_rag.chunking import Chunk, chunk_corpus, relevant_chunk_ids
from ailab_rag.config import load_config
from ailab_rag.data import Question, load_corpus, load_questions
from ailab_rag.eval import main as eval_main
from ailab_rag.providers import StubProvider
from ailab_rag.reader import Reader
from ailab_rag.retrieval import BM25


def _pick(questions: Sequence[Question], *, answerable: bool) -> Question | None:
    return next((q for q in questions if q.answerable is answerable), None)


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config()
    docs = load_corpus(config.corpus)
    questions = load_questions(config.questions)
    chunks = chunk_corpus(docs, config.chunk_size, config.chunk_overlap)
    text_by_id = {c.id: c.text for c in chunks}
    bm25 = BM25(chunks)
    reader = Reader(StubProvider(), support_threshold=config.support_threshold)
    k = config.top_k

    print("== ailab-rag demo ==\n")
    print(
        f"corpus: {len(docs)} docs -> {len(chunks)} chunks "
        f"(chunk_size={config.chunk_size} tok, overlap={config.chunk_overlap} tok), top_k={k}\n"
    )

    picks = (_pick(questions, answerable=True), _pick(questions, answerable=False))
    samples = [q for q in picks if q]
    for q in samples:
        print(f"Q [{q.id}] ({'answerable' if q.answerable else 'unanswerable'}): {q.question}")
        scored = bm25.scored(q.question, k)
        for rank, (cid, score) in enumerate(scored, start=1):
            print(f"    {rank}. {cid}  score={score:.3f}")
        retrieved = [(cid, text_by_id[cid]) for cid, _ in scored]
        if retrieved:
            top_id, top_text = retrieved[0]
            preview = " ".join(top_text.split()[:18])
            print(f"    context[0] {top_id}: {preview} ...")
        result = reader.answer(q.question, retrieved)
        if result.refused:
            print(f"    -> REFUSED: {result.answer}\n")
        else:
            print(f"    -> ANSWER: {result.answer}")
            print(f"       CITES: {', '.join(result.citations) or '(none)'}\n")

    _worked_example(questions, chunks, bm25, k)

    print("== eval ==\n")
    return eval_main(argv)


def _worked_example(questions: Sequence[Question], chunks: list[Chunk], bm25: BM25, k: int) -> None:
    """Find and print one relevant chunk ranked beyond the top-k cutoff."""
    deep = len(chunks)
    for q in questions:
        if not q.answerable:
            continue
        rel = relevant_chunk_ids(q, chunks)
        if not rel:
            continue
        ranked = bm25(q.question, deep)
        for rank, cid in enumerate(ranked, start=1):
            if cid in rel and rank > k:
                print("== worked example: a relevant chunk excluded by the cutoff ==")
                print(
                    f"Q [{q.id}]: {q.question}\n"
                    f"    relevant chunk {cid} is at rank {rank}, but top_k={k}, "
                    f"so it is EXCLUDED. Raising top_k to {rank} would retrieve it.\n"
                )
                return


if __name__ == "__main__":
    raise SystemExit(main())
