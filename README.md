# AURORA

ABN AMRO × AISO project. Local PageIndex RAG over the Writing Guide and EN articles.

## Layout

```
AURORA/
├── data/        # source PDFs + EN/NL articles
├── docs/        # project documents and diagrams
├── rag/         # retrieval + indexing (local PageIndex, no hosted API)
│   ├── pageindex/   # vendored PageIndex library
│   ├── scripts/     # build_corpus.py, run_pageindex.py
│   ├── corpus/      # generated corpus + tree JSONs
│   ├── notebooks/   # aurora_demo.ipynb, crash_course.ipynb
│   ├── examples/    # upstream reference examples
│   └── legacy/      # archived pre-reorg material
├── backend/     # (placeholder)
└── frontend/    # (placeholder)
```

## Setup

```bash
uv sync                  # creates .venv, installs deps
cp .env.example .env     # add your OPENAI_API_KEY
```

## Build the EN corpus

```bash
uv run python rag/scripts/build_corpus.py
```

Writes `rag/corpus/corpus_en.md`.

## Index a document (local, no hosted API)

```bash
uv run python rag/scripts/run_pageindex.py --pdf_path data/"Writing Guide 2026-V1.1.pdf"
uv run python rag/scripts/run_pageindex.py --md_path rag/corpus/corpus_en.md
```

Output JSON lands in `./results/`.

## Notebooks

```bash
uv run jupyter lab rag/notebooks/
```

`aurora_demo.ipynb` is the end-to-end demo. `crash_course.ipynb` is the PageIndex tutorial.
