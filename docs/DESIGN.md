# Design notes

Decisions behind `ailab-rag`, and the alternatives that were considered and rejected. This
lab replaces the classification seam of the template with a retrieve-then-read RAG
pipeline, and keeps the template's discipline: a number gates CI.

## Provenance

This lab was scaffolded from two sources, both at `origin/main`:

- `ailab-template-python` at `00c298d931c8452ad1bfe9a42a3e7e23998f7c1f`: the eval runner,
  regression gate, config layering, secret wall (pre-push hook plus CI), non-root Docker,
  ruff, mypy, pytest with a coverage floor, and the `eval_results.json` `schema_version`
  contract.
- `ailab-evals` at `8c3ff14597b462fd927e1ac0c16f66c09392c586`: the retrieval metric math
  (recall@k, hit@k, MRR, nDCG@k) and the BM25 approach, ported verbatim.

The SHAs are also recorded in `BUILD_SOURCES.txt` and `FIXTURES.md`.

## 1. The seam is retrieve-then-read, behind small Protocols

The template's `Classifier` seam is replaced with a `Retriever` protocol
(`fn(query, k) -> ranked chunk ids`) and a `Reader` seam behind the existing `LLMProvider`.
The eval scores whatever satisfies the protocol.

- Rejected: a single end-to-end answerer. Splitting retrieval from reading lets each half
  be measured on its own, which is the whole point of a lab.
- Rejected: a framework (LangChain, LlamaIndex). Heavy dependency trees for two methods,
  and a vendor choice every lab would inherit. This lab keeps `dependencies = []`.

## 2. BM25 is pure Python and the metric math is ported verbatim

BM25 runs over chunk texts with no dependency. The four ranking functions are copied
character for character from `ailab-evals`, and `tests/test_metric_parity.py` asserts the
copies still agree with the originals.

- Rejected: re-deriving the metrics. A re-derivation can disagree subtly with the source.
  A verbatim copy plus a parity test is safer and is the operator's explicit instruction.
- Rejected: a vector or embedding retriever in v1. That needs a model and a dependency.
  Dense retrieval is a later slice behind the same `Retriever` seam.

## 3. Relevance is by span containment, not by chunk id

A chunk is relevant to a question when its text contains the question's answer span, after
lowercasing and whitespace normalisation. Relevance is recomputed for each chunk set.

- Rejected: labelling relevance by a fixed chunk id. Chunk ids move when `chunk_size`
  changes, so id-based labels would rot. Span containment keeps the gold labels valid
  across chunk sizes, which is what makes the chunk-size experiment honest.

## 4. The chunk unit is whitespace tokens, documented and consequential

`chunk_size` and `chunk_overlap` are counted in whitespace tokens. Changing them rebuilds
the chunk set. The lab proves this matters: question `q14` has a fact that spans a 16-token
window boundary, so it is unretrievable at `chunk_size` 16 and retrievable at 32.

- Rejected: character or sentence chunking for v1. Tokens are simple, deterministic, and
  enough to show the effect. A real system would use a subword tokenizer.

## 5. The gate uses a tolerance band and several honesty checks

The gate fails below `floor - tolerance`, not on an exact score, so it does not trip on
noise. It also fails on a saturated corpus, on baselines it cannot beat, on a fabricated
answer to an unanswerable question, and on a citation that does not support its answer.

- Rejected: an exact threshold. A single point score flaps with one example on a small set.
- Rejected: comparing to the previous run. That needs stored state and lets quality ratchet
  down in small steps. Explicit floors force a visible diff in review.

## 6. The generation check is named honestly

The stub reader's check is `citation_resolvable`, not `faithfulness`. It proves a cited id
exists, nothing more. A deterministic `lexical_support` check scores n-gram overlap with the
cited chunk, with a negative control in the tests. The report states that stub-mode is a
FORMAT path, not a real model.

- Rejected: calling the stub check faithfulness. That would overclaim. Faithfulness needs a
  model that reads for meaning, which is the opt-in recorded live pass.

## 7. The live provider is record-and-replay, not live in CI

`OllamaProvider` records real completions once. `ReplayProvider` serves a committed JSON
cassette keyed by `sha256(model, prompt)`. A cassette miss raises, so a stale cassette
cannot silently pass.

- Rejected: a live model in CI. Non-deterministic, costs money or needs a server, and turns
  an outage into a red build.

## 8. Kept from the template

The secret wall (pre-push hook plus a required CI check, both scanning full history), the
non-root multi-stage Docker image, uv with a committed lockfile, strict mypy, a 90 percent
coverage floor, and the versioned `eval_results.json` contract. These are not re-argued
here; see the template's own design notes.

## 9. Content-rule enforcement

`make lint-docs` fails on the em dash character (U+2014) anywhere in the docs.
`make check-learning-numbers` fails if a tagged figure in `docs/LEARNING.md` differs from
the committed results. Both run in CI, so a hand-typed or drifted number cannot merge.

## 10. Things deliberately left out of v1

Dense or embedding retrieval, reranking, fine-tuning, caching, a browser UI, and
multi-agent orchestration. Each is a later slice behind the seam this lab already defines.
