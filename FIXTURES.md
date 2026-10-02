# Fixture provenance

Every file under `fixtures/` is listed here. A fixture without an entry must not be merged.

## Rule

**Only public or synthetic data may ever be added to this repository.** That means:

- no real customer, user or employee data, and no personal information of any kind;
- no internal business data, logs, tickets, metrics or exports, even if anonymised;
- no hostnames, IP addresses, account names or other infrastructure identifiers;
- public datasets only under a license that permits redistribution, with the source,
  version and license recorded below.

Synthetic text is written fresh for the fixture, not adapted from real records. The secret
wall (`make scan`) catches some mistakes; it cannot catch all of them, so this rule is
enforced in review as well.

## Source commits

This lab ported its retrieval math and scaffold from two repositories at `origin/main`:

- `ailab-template-python` at `00c298d931c8452ad1bfe9a42a3e7e23998f7c1f`
- `ailab-evals` at `8c3ff14597b462fd927e1ac0c16f66c09392c586`

## Files

### `fixtures/nimbus_corpus.jsonl`

| Field | Value |
|---|---|
| sha256 | `1350f49a0522dc164f0bf01daa560d310f0e2b5faf79d64bea99c071bc7d9c8d` |
| Rows | 20 documents |
| Format | JSONL, one `{"id", "title", "text"}` object per line |
| World | A fictional cloud backup and sync SaaS named Nimbus (overview, runbooks, API, security, pricing, troubleshooting) |
| Author | Andrew Pyle, written by hand for this lab (2026-10-01), with AI assistance |
| Origin | Synthetic. Not derived from, sampled from, or paraphrased from any real product, docs, or support data |
| Personal data | None. No names, emails, account numbers, or identifiable details |
| Business data | None. The product Nimbus does not exist |
| License | Apache-2.0, same as the repository |
| Known bias | Single author, short and clean English; facts are placed in later paragraphs so the lead baseline is weak |

### `fixtures/nimbus_questions.jsonl`

| Field | Value |
|---|---|
| sha256 | `5067828f57a1e0eee1edc6099c683a320c5d36004b7fb8e847a16302fb08a636` |
| Rows | 18 questions (14 answerable, 4 unanswerable) |
| Format | JSONL, one `{"id", "question", "doc_id", "answer_span", "answerable", "critical", "trap"}` object per line |
| Ground truth | Span-keyed. Each answerable question stores the exact answer span in its source document. Relevance is by span containment per chunk set |
| Traps | 3 answerable questions marked `trap` share vocabulary or a number with a distractor document (for example a status code the doc also mentions) |
| Boundary split | `q14` has an 18-token answer span that splits across a 16-token chunk window, so it is unretrievable at `chunk_size` 16 and retrievable at 32 |
| Unanswerable | 4 near-miss questions share topic vocabulary (storage plan, version history, cipher, webhook retry) but each carries a word absent from the corpus (for example "Enterprise", "quantum", "millisecond"), so no answer exists |
| Critical | 5 questions marked `critical` carry a `critical recall` subset line |
| Author | Andrew Pyle, written by hand for this lab (2026-10-01), with AI assistance |
| Origin | Synthetic. No real questions or support transcripts |
| License | Apache-2.0 |

### `fixtures/cassettes/nimbus_reader.json`

| Field | Value |
|---|---|
| sha256 | `2f75874cca890f0c38c62d9d17a7b361359b4910210914b84863b0b5109903f2` |
| Format | JSON cassette keyed by `sha256(model, prompt)`, with `{"model", "response"}` entries |
| Purpose | Lets `ReplayProvider` replay Reader completions with no network, exercised by `tests/test_providers.py` |
| Origin | SYNTHETIC. These entries were recorded from the deterministic offline `StubProvider` as a stand-in, NOT from a live model. One real recording with `OllamaProvider` on a Mac with Ollama remains to be run and committed |
| License | Apache-2.0 |
