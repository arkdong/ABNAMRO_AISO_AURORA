# Advanced chunking methods to try

Beyond the baseline strategies in [`experiments.md`](experiments.md) (C1–C10),
these are more research-flavored chunkers worth evaluating. Each entry lists
what it is, why it's interesting for AURORA, and the rough cost.

When one of these graduates from "want to try" to "actually testing", give it
an ID (`A1`, `A2`, …) and add it to the results table in `experiments.md`.

---

## A1 — Contextual retrieval (Anthropic-style)

Before embedding, prepend a short LLM-generated blurb to each chunk that
explains where it sits in the document ("This chunk is from the 2025
agriculture outlook, section on dairy margins").

- **Why interesting:** Anthropic reported ~35–50% retrieval error reduction.
  Plays directly to our Claude stack, and prompt caching makes it cheap.
- **Cost:** one LLM call per chunk at index time (cached). No extra cost at
  query time.
- **Status:** _candidate_.

## A2 — Late chunking (Jina)

Embed the whole document at once with a long-context embedder, then
mean-pool over chunk spans afterwards. Each chunk's vector still "knows
about" the surrounding document.

- **Why interesting:** removes the cross-chunk context loss that vanilla
  chunking introduces. No LLM cost at all.
- **Cost:** needs a long-context embedder (e.g. `jina-embeddings-v3`).
  Slower per-doc indexing, no inference change.
- **Status:** _candidate_.

## A3 — Proposition-based chunking

LLM decomposes text into atomic, self-contained claims — one per chunk
("Dairy margins fell 12% in Q3 2024 due to feed costs").

- **Why interesting:** every chunk stands alone, no coreference back to
  surrounding text. Strong for factual queries.
- **Caveat:** weaker for *stylistic* retrieval, so probably better for the
  Writing Guide than for article style references.
- **Cost:** one LLM call per chunk at index time.
- **Status:** _candidate_.

## A4 — RAPTOR (hierarchical summary tree)

Cluster chunks, summarize each cluster with an LLM, recurse upward into a
tree. Index both originals AND summaries; retrieval can hit any level.

- **Why interesting:** handles queries at different granularities — a
  specific fact vs. "what does ABN's stance on agriculture look like
  overall."
- **Cost:** non-trivial. Multiple LLM calls per cluster, plus recursive
  re-indexing. Probably overkill for a PoC.
- **Status:** _stretch goal._

## A5 — Small-to-big / parent-document retrieval

Embed small chunks (1–2 sentences) for precise matching, but return the
surrounding paragraph or section at retrieval time.

- **Why interesting:** sharp recall on the small embeddings, full context
  for the generation LLM. Cheap to implement, often a clear precision win.
- **Cost:** double storage (small chunks indexed, parent text stored
  alongside). No extra LLM calls.
- **Status:** _candidate, low effort — probably try this early._

## A6 — HyDE-style indexing (question generation per chunk)

For each chunk, an LLM generates 3–5 questions the chunk would answer.
Embed the questions and link them back to the chunk.

- **Why interesting:** content-writer prompts are often question-shaped
  ("how does ABN frame nitrogen rules?"). Embedding question-to-question is
  often stronger than question-to-passage.
- **Cost:** LLM calls per chunk at index time, plus 3–5× more vectors.
- **Status:** _candidate_.

## A7 — LLM-as-chunker

Give the LLM the full document and ask it to split at semantically coherent
boundaries, with rationale.

- **Why interesting:** useful as a *ceiling* baseline — if cheap chunkers
  approach this, we know we've hit diminishing returns.
- **Cost:** expensive and slow. Only worth running once per document.
- **Status:** _baseline only._

## A8 — Structure-aware (Writing Guide specifically)

Parse the Writing Guide PDF's heading hierarchy (H1 > H2 > H3) into a tree.
Chunk by leaf rule, but carry the full breadcrumb path as metadata so we
can filter retrieval to "rules under section 3.x".

- **Why interesting:** the Writing Guide is highly structured. Lets us do
  cheap structured filtering at query time.
- **Cost:** PDF parsing complexity; one-time effort.
- **Status:** _candidate, complementary to other strategies (it's about
  metadata, not chunk shape)._

---

## Suggested ordering

If we can only run a few of these inside the MVP timeline:

1. **A5 — small-to-big.** Cheapest with consistent gains.
2. **A1 — contextual retrieval.** Best expected ROI, plays to our stack.
3. **A8 — structure-aware** for the Writing Guide.
4. **A6 — HyDE-style** if query-shape mismatch is hurting precision.
5. **A2, A3, A4, A7** as research stretch goals if time allows.
