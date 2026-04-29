# Advanced embedding techniques to try

These sit *on top of* a base embedding model (the E1–E9 list in
[`experiments.md`](experiments.md)) — they're wrappers and augmentations,
not replacements. Picking a base model decides ~5–10% of retrieval
quality; these techniques typically decide more.

When one of these graduates from "want to try" to "actually testing", give
it an ID (`X1`, `X2`, …) and add it to the results table in `experiments.md`.

---

## Tier 1 — cheap, usually big wins

### X1 — Cross-encoder reranking

Vector search returns top-50; re-score them with a cross-encoder
(`BAAI/bge-reranker-v2-m3`, Cohere Rerank, or `jina-reranker-v2`) and keep
the new top-5/10.

- **Why interesting:** typically the single biggest quality jump in a RAG
  pipeline. Cross-encoders see query and passage *together*, which dense
  embeddings can't.
- **Cost:** +50–200ms per query. No extra storage.
- **Expected gain:** +10–20% precision@5.
- **Status:** _candidate, try first._

### X2 — HyDE at query time

Use an LLM to generate a hypothetical *answer* to the query, then embed
that answer instead of (or in addition to) the raw query.

- **Why interesting:** fixes question-vs-passage asymmetry. Real queries
  ("how does ABN frame nitrogen rules?") don't look like passages, but a
  generated fake answer does.
- **Cost:** one LLM call per query. Cacheable.
- **Expected gain:** +3–10%, more on question-shaped queries.
- **Status:** _candidate._

### X3 — Multi-query + Reciprocal Rank Fusion

LLM rewrites the query 3–5 ways, retrieve for each, fuse the result lists
with RRF.

- **Why interesting:** robust to query phrasing variance, plays well with
  every other technique here.
- **Cost:** N× retrievals + 1 LLM call per query.
- **Expected gain:** +5–10%.
- **Status:** _candidate._

### X4 — Hybrid BM25 + dense

Run BM25 keyword search and dense vector search in parallel, fuse with
RRF or weighted scores.

- **Why interesting:** BM25 catches exact-term matches dense embeddings
  blur — sector names, "ABN AMRO", "Basel III", "stikstofcrisis". Most
  serious production RAG does this.
- **Cost:** maintain a BM25 index alongside Chroma. No LLM cost.
- **Expected gain:** +5–15%, biggest on jargon-heavy queries.
- **Status:** _candidate._

### X5 — Instruction prefixes (don't screw this up)

Models like e5 and BGE expect `"query: ..."` and `"passage: ..."` prefixes
at inference. Forgetting them silently drops precision by 5–15%.

- **Why call out:** not really a technique — more a footgun. Worth a
  dedicated check in the embedding pipeline.
- **Cost:** zero.
- **Status:** _do this from day one._

---

## Tier 2 — research-flavored, more interesting

### X6 — ColBERT / late interaction

Token-level multi-vector embeddings; scoring uses MaxSim across all
token pairs.

- **Why interesting:** stronger on long passages than single-vector dense.
  Particularly good when chunks contain multiple distinct facts.
- **Cost:** 10–50× more vectors stored. Needs a ColBERT-aware index
  (Chroma doesn't natively support it; would need PLAID or Qdrant).
- **Status:** _stretch goal._

### X7 — BGE-M3 multi-output (dense + sparse + ColBERT)

BGE-M3 emits three representations from one forward pass: dense vector,
learned sparse (lexical) vector, and ColBERT-style multi-vector. Run all
three retrievals and fuse.

- **Why interesting:** "free" three-way retrieval since one model produces
  all three. Closest thing to a one-stop shop for hybrid retrieval.
- **Cost:** 3× storage and retrieval, but one model load.
- **Status:** _candidate, high research value._

### X8 — Matryoshka (MRL) embeddings

Embeddings trained so a truncated slice (e.g. first 256 of 1024 dims)
still works. Allows two-pass retrieval: fast low-dim first pass, full-dim
re-score on top-N.

- **Why interesting:** retrieval speed/storage tradeoff dial. OpenAI
  `text-embedding-3-large` supports this natively, as do some BGE
  variants.
- **Cost:** none beyond base embedding.
- **Status:** _candidate, try when corpus grows._

---

## Tier 3 — stretch / out-of-scope-ish

### X9 — Cross-lingual alignment fine-tune

Fine-tune the embedder so Dutch queries and English passages (or vice
versa) land close in vector space.

- **Why interesting:** only worth it if Q1 (translate-before-embed vs raw
  Dutch) lands on "raw NL is better, but cross-lingual retrieval is
  weak."
- **Cost:** real fine-tuning effort + labeled pairs.
- **Status:** _conditional on Q1 outcome._

### X10 — Domain fine-tuning on banking content

Lightly fine-tune the chosen embedder on contrastive pairs from real
writer (query, relevant chunk) data.

- **Why interesting:** can dominate generic models for niche domains.
- **Cost:** needs labeled pairs (which we'll have from the eval set
  anyway). MVP scope says no, but it's the obvious next step post-MVP.
- **Status:** _post-MVP._

### X11 — Binary / int8 quantization

Store embeddings as binary or int8 instead of float32. ~32× storage and
speed gain for ~3% recall hit.

- **Why interesting:** matters in production at Azure scale; irrelevant
  for a local PoC.
- **Cost:** small recall loss, no extra compute.
- **Status:** _post-MVP._

---

## Stacking notes

These compose. A reasonable "everything wired up" pipeline:

1. Query → multi-query rewrite (X3) + HyDE (X2) → N expanded queries
2. Each expanded query → embed with chosen base model (with correct
   prefix, X5) → dense retrieval
3. In parallel: BM25 retrieval over the same corpus (X4)
4. Fuse all candidate sets with RRF
5. Cross-encoder rerank top-50 (X1) → final top-5/10

That's roughly state-of-the-art for non-fine-tuned RAG.

## Suggested ordering for MVP

1. **X5 — instruction prefixes.** Not optional; catches a silent bug.
2. **X1 — reranker.** Highest ROI single technique.
3. **X4 — hybrid BM25 + dense.** Cheap, big impact on jargon queries.
4. **X2 — HyDE.** Easy to add once an LLM call is already in the pipeline.
5. **X3 — multi-query / RRF.** Adds robustness once the rest is in place.
6. **X7 — BGE-M3 multi-output.** Most interesting research piece.
7. **X6, X8, X9, X10, X11** — stretch / post-MVP.
