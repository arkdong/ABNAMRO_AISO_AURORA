# Research questions

Open questions feeding the comparative analysis deliverable.
Update each one with findings as experiments produce data.

## Q1 — Translate before embedding, or embed Dutch directly?

Does translating Dutch articles to English before embedding improve retrieval
compared to using a multilingual model on raw Dutch text?

- **Why it matters:** translation introduces noise and lossy paraphrase, but
  unifies article corpus with the English Writing Guide.
- **How we test:** run E1/E4/E6 (multilingual) over both T1 (translated) and
  T2 (raw Dutch). Compare P@5 on the same eval queries.
- **Status:** not started.
- **Finding:** _tbd_

## Q2 — Best chunking strategy for insight articles?

- **Why it matters:** chunk boundaries shape what the model can retrieve.
  Fixed-size is simple but cuts mid-sentence; paragraph respects structure
  but produces uneven sizes; semantic is theoretically best but slowest.
- **How we test:** C1 vs C4 vs C5 vs C6 with the same embedder (E1).
- **Status:** not started.
- **Finding:** _tbd_

## Q3 — Best embedding model for mixed NL/EN content?

- **Why it matters:** the corpus is bilingual; English-only models will fail
  on raw Dutch, but they may dominate translated content.
- **How we test:** lock chunker to the Q2 winner, sweep E1–E9.
- **Status:** not started.
- **Finding:** _tbd_

## Q4 — Optimal top-k?

At what top-k does adding more retrieved chunks stop improving generation
quality?

- **Why it matters:** spec defaults are k=3 for articles, k=10 for rules,
  but these are guesses.
- **How we test:** sweep k ∈ {1, 3, 5, 10, 15, 20} once generation is wired,
  judge generated articles for compliance + style fit.
- **Status:** blocked on generation step.
- **Finding:** _tbd_

## Q5 — Should writing-guide chunks carry parent-section context?

C8 attaches the parent section heading to each rule chunk. Does this improve
retrieval over plain C7?

- **Status:** not started.
- **Finding:** _tbd_
