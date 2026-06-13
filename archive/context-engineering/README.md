# context-engineering

RAG retrieval + generation for the AURORA editorial co-pilot. Given a user
query, this module retrieves (a) similar prior insight articles as **style
references** and (b) relevant chunks of ABN AMRO's Writing Guide as **rules**,
composes a structured prompt, and (optionally) calls an LLM to generate the
article.

This is the **Track A — Vector RAG** implementation. The parallel
**Track B — PageIndex / vectorless** lives in [`../rag/`](../rag).

---

## TL;DR — using from Python

```python
from scripts import RAG

rag = RAG()                                          # defaults baked in
bundle = rag.retrieve("How is AI changing advertising for Dutch businesses?")

bundle.style_references   # list[RetrievedChunk]  — relevant prior articles
bundle.writing_rules      # list[RetrievedChunk]  — relevant WG sections
bundle.composed_prompt    # str                   — assembled USER message
bundle.debug              # dict                  — config + timings

# Full pipeline (retrieve → compose → call LLM → return article):
article = rag.generate(
    "How is AI changing advertising for Dutch businesses?",
    llm_provider="openai",   # or "anthropic"
)
```

That's the full integration surface. The CLI in
[`scripts/generate.py`](scripts/generate.py) is a thin wrapper around the
same `RAG` class.

---

## The return shape

```python
@dataclass
class ContextBundle:
    query: str
    style_references: list[RetrievedChunk]
    writing_rules:    list[RetrievedChunk]
    composed_prompt:  str                    # the assembled USER message
    debug:            dict                   # config + timings

@dataclass
class RetrievedChunk:
    text:     str
    metadata: dict                           # see below
    score:    float                          # rerank → rrf → bm25 → similarity
```

Common `metadata` keys:

| Field | Articles | Writing Guide |
|---|---|---|
| `source_title` | article title | — |
| `source_date` | YYYY-MM-DD | — |
| `sector` | e.g. `"Technologie, Media & Telecom"` | — |
| `source_slug` | filename slug | — |
| `breadcrumb` | — | `"3. Wording > 3.1 Style > Put the reader centre stage"` |
| `section_title` | — | leaf section title |
| `depth` | — | 1 / 2 / 3 (chapter / section / rule) |
| `node_id` | — | stable node identifier |
| `chunk_id`, `chunker`, `language_embedded` | always | always |

---

## Setup

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. API keys (optional — only needed for live LLM calls and a3/a4/a6/a7
#    LLM-based chunkers; dry-run + all retrieval works without keys)
cp .env.example .env       # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY

# 3. Extract the writing guide and build chunks + indexes (one-time)
python -m scripts.ingest_pdf                                          # PDF → markdown
python -m scripts.chunk --method a9  --src gpt-5                      # article chunks
python -m scripts.chunk --method a10 --src writing-guide              # WG chunks
python -m scripts.embed --embedder e4 --chunker a9  --src gpt-5       # dense vectors
python -m scripts.embed --embedder x4 --chunker a9  --src gpt-5       # BM25 (articles)
python -m scripts.embed --embedder x4 --chunker a10 --src writing-guide  # BM25 (WG)

# 4. Sanity check (no API key needed)
python -m scripts.generate \
    --query "How is AI changing advertising for Dutch businesses?" \
    --dry-run
```

Indexes land in `vector_db/` (gitignored — large binaries, reproducible from
`chunked/*.jsonl`). Re-running step 3 is idempotent; pass `--force` to rebuild.

---

## CLI

```bash
# Dry-run: prints composed prompt + retrieved context, no LLM call
python -m scripts.generate --query "..." --dry-run

# Real generation
export OPENAI_API_KEY=sk-...
python -m scripts.generate --query "..." --llm-provider openai --model gpt-4o

# Inspect what was retrieved + the full prompt the LLM sees
python -m scripts.generate --query "..." --show-context --show-prompt
```

All flags: `python -m scripts.generate --help`.

A scratch runner with the canonical setup commands is in
[`scripts.sh`](scripts.sh).

---

## The default recipe

Defaults are encoded as `DEFAULT_*` constants in
[`scripts/api.py`](scripts/api.py). They mirror the best-performing stack
from the experiments under [`experiments/`](experiments).

| Component | Default | Why |
|---|---|---|
| **Embedder** | `e4` — BGE-M3 | Multilingual, no prefix required, strong on NL + EN |
| **Article chunker** | `a9` — hybrid small-to-big | Semantic-parent + sentence-window children: reranker scores tight children, LLM gets full-context parents |
| **Writing-guide chunker** | `a10` — RAPTOR-structural | 3-level tree (chapter → section → rule); deterministic, no LLM at build time |
| **Retrieval mode** | hybrid (dense + BM25 via RRF k=60) | Catches both semantic similarity and keyword matches |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Cross-encoder; multilingual; same family as the embedder |
| **Top-K articles** | 3 | Balance between context breadth and prompt token cost |
| **Top-K rules** | 5 | |
| **Top-N (candidates per retriever)** | 30 | Recall buffer before the reranker narrows |

Every default is a runtime knob — override via `RAG(...)` constructor or
matching CLI flag.

---

## Configuration

### Per-call overrides

```python
bundle = rag.retrieve(
    "AI in advertising",
    top_k_articles=5,        # default 3
    top_k_rules=8,           # default 5
)

article = rag.generate(
    "AI in advertising",
    llm_provider="openai",
    model="gpt-4o-mini",     # override default model
    max_tokens=2048,
)
```

### Constructor knobs

```python
rag = RAG(
    embedder="e4",
    articles_chunker="a9",
    wg_chunker="a10",
    articles_src="gpt-5",          # translation source folder name
    wg_src="writing-guide",
    top_n=30,
    top_k_articles=3,
    top_k_rules=5,
    use_reranker=True,             # disable for fast iteration
    reranker_model="BAAI/bge-reranker-v2-m3",
    expand_parents=True,           # small-to-big parent expansion for a5/a9
    system_prompt=SYSTEM_PROMPT,   # override if you need a different role
)
```

### One-corpus retrieval

If you only need articles or only need rules:

```python
refs  = rag.retrieve_style_references("...", top_k=5)
rules = rag.retrieve_writing_rules("...", top_k=10)
```

---

## Architecture

```
../../data/                            (shared repo-root data)
├── article/{en,nl}/*.md               source corpus (NL + GPT-5 translations)
├── writing_guide.md                   extracted by scripts/ingest_pdf.py
└── Writing Guide 2026-V1.1.pdf        source PDF

context-engineering/                   (this folder)
├── scripts/
│   ├── api.py                         importable Python API (RAG, ContextBundle)
│   ├── generate.py                    CLI wrapper around api.py
│   ├── chunk.py                       21 chunking strategies (c1-c11, a1-a10)
│   ├── embed.py                       10 dense embedders + BM25 + BGE-M3 multi
│   ├── retrieve.py                    dense / BM25 / hybrid + rerank + expand
│   ├── experiment.py                  experiment harness
│   └── ingest_pdf.py                  PDF → markdown for the Writing Guide
├── chunked/                           pre-computed JSONL chunks per (chunker × src)
├── vector_db/                         Chroma + BM25 indexes per embedder (gitignored)
├── experiments/                       experiment outputs per run-id (gitignored)
└── tests/
```

### Runtime pipeline

```
                   ┌──────────────────────────────┐
   user query  ──▶ │ retrieve_style_references   │  hybrid + rerank + parent-expand
                   │   articles via A9 + E4 + X4  │
                   └────────────────┬─────────────┘
                                    │  list[RetrievedChunk]
                                    │
                   ┌────────────────▼─────────────┐
   user query  ──▶ │ retrieve_writing_rules      │  hybrid + tree-expand + rerank
                   │   WG via A10 + E4 + X4       │
                   └────────────────┬─────────────┘
                                    │  list[RetrievedChunk]
                                    ▼
                   ┌──────────────────────────────┐
                   │ compose_user_message         │  fixed template:
                   │                              │  RULES + REFERENCES + TASK
                   └────────────────┬─────────────┘
                                    │  str
                                    ▼
                   ┌──────────────────────────────┐
                   │ call_anthropic / call_openai │  Anthropic uses prompt caching
                   └────────────────┬─────────────┘
                                    │
                                    ▼
                            generated article
```

---

## Build-time alternatives

The defaults are the recommended stack. The chunk/embed pipeline supports
many alternatives for experimentation:

### Chunkers ([`scripts/chunk.py`](scripts/chunk.py))

| Tier | ID | Strategy | Target |
|---|---|---|---|
| simple | c1 / c2 / c3 | fixed-size 512/256/1024 with overlap | articles |
| simple | c4 | paragraph | articles |
| simple | c5 | sentence-window (focal sentence ± neighbours) | articles |
| simple | c6 | semantic (split where adjacent-sentence similarity drops) | articles |
| simple | c10 | article-as-chunk | articles |
| simple | c7 / c8 / c9 / c11 | heading-based variants | writing guide |
| advanced | a1 | contextual retrieval (Anthropic-style; needs LLM key) | articles |
| advanced | a2 | late chunking | articles |
| advanced | a3 | proposition-based (LLM) | articles |
| advanced | a4 | RAPTOR (2-level summaries; LLM) | articles |
| advanced | a5 | small-to-big | articles |
| advanced | a6 | HyDE-style indexing (LLM) | articles |
| advanced | a7 | LLM-as-chunker (LLM) | articles |
| advanced | a8 | structure-aware | writing guide |
| advanced | a9 | hybrid small-to-big (semantic parents + sentence-window children) | articles |
| advanced | a10 | RAPTOR-structural (3-level tree) | writing guide |

Build them all at once:

```bash
python -m scripts.chunk --method all-no-key --src gpt-5
python -m scripts.chunk --method all-no-key --src writing-guide
```

### Embedders ([`scripts/embed.py`](scripts/embed.py))

| ID | Model | Dim | Notes |
|---|---|---|---|
| e1 | intfloat/multilingual-e5-large | 1024 | needs `passage:` / `query:` prefix |
| e1i | intfloat/multilingual-e5-large-instruct | 1024 | uses `Instruct: ... Query:` |
| e2 | intfloat/multilingual-e5-base | 768 | needs prefix |
| e3 | sentence-transformers/all-MiniLM-L6-v2 | 384 | English-only |
| e4 | BAAI/bge-m3 | 1024 | **default** |
| e5 | BAAI/bge-large-en-v1.5 | 1024 | English-only |
| e6 | jinaai/jina-embeddings-v3 | 1024 | multilingual |
| e7 | openai/text-embedding-3-large | 3072 | API; Matryoshka-capable |
| e8 | openai/text-embedding-3-small | 1536 | API; Matryoshka-capable |
| e9 | cohere/embed-multilingual-v3.0 | 1024 | API |
| x4 | BM25 keyword index | — | sparse; powers hybrid retrieval |
| x7 | BGE-M3 multi-output (dense + sparse) | 1024 | single forward pass for both |
| x8 | Matryoshka truncation flag | varies | use with `--matryoshka-dim N` |

Build everything that doesn't need an API key:

```bash
python -m scripts.embed --embedder all-no-key --chunker a9 --src gpt-5
```

---

## Experimentation

[`scripts/experiment.py`](scripts/experiment.py) runs a set of pre-defined
experiments side-by-side and writes per-experiment artifacts (config,
retrieved chunks, composed prompt, LLM output, summary) into
`experiments/<run-id>/<experiment-name>/`.

```bash
python -m scripts.experiment --list                  # show defined experiments
python -m scripts.experiment                         # run all
python -m scripts.experiment --only baseline,no_rerank
python -m scripts.experiment --dry-run               # skip LLM calls
python -m scripts.experiment --compare latest        # comparison table
```

Add a new experiment by appending a dict to the `EXPERIMENTS` list at the
top of `experiment.py`.

---

## Testing

```bash
python -m pytest tests/
```

Coverage at this stage is minimal — see [`tests/`](tests).

---

## Data assumptions

Paths are resolved relative to the **repo root**, not this folder.

- `data/article/en/*.md` — translated insight articles with YAML frontmatter
- `data/article/nl/*.md` — Dutch originals
- `data/writing_guide.md` — extracted by [`scripts/ingest_pdf.py`](scripts/ingest_pdf.py)
- `data/Writing Guide 2026-V1.1.pdf` — source PDF

Articles are translated by [`../document_processing/`](../document_processing).

---

## Known limitations

- `vector_db/` is **not committed** — every contributor builds locally
  (`scripts.embed`). First build downloads the embedder model (~1–2 GB for BGE-M3).
- Reranker is the dominant query-time cost (~7–20s per query on CPU). Disable
  with `--no-rerank` for fast iteration; re-enable for quality. The latency hit
  comes from the cross-encoder model loading on first invocation.
- Current scope is the **TMT sector** corpus only. Adding sectors requires
  extending [`../document_processing/`](../document_processing) with new
  scrapers and re-running chunk + embed.
- Article chunker defaults to `a9`; writing-guide chunker defaults to `a10`.
  Switching either requires a matching `scripts.embed` run for the new
  (chunker × src) collection.

---

## See also

- [`../docs/comparison_matrix.md`](../docs/comparison_matrix.md) — Track A (this code) vs Track B (vectorless) evaluation framework
- [`../docs/project_overview.md`](../docs/project_overview.md) — full project context
- [`../rag/`](../rag) — Track B (vectorless PageIndex retrieval)
- [`../profiles/`](../profiles) — workflow + domain-expert profile registry
- [`../task_definition/`](../task_definition) — intent classification (T1_DRAFT, T1_TRANSLATE, T1_SEARCH, T2_COMPLIANCE, T4_RENEWAL)
