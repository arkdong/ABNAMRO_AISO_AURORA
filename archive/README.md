# AURORA Archive

AURORA Archive is the older local proof-of-concept for an editorial co-pilot.
The archived demo takes a
writing request, classifies the intent, selects relevant workflow and expert
profiles, retrieves supporting context from ABN AMRO writing assets, refines the
prompt with clarification questions, generates draft content, and evaluates the
result against a KPI catalogue.

The archived project is contained in this `archive/` folder. Run commands from this
directory unless a command says otherwise.

## What The Demo Does

- Runs a Streamlit chat UI for the end-to-end AURORA pipeline.
- Uses cached PageIndex trees over the repo-root `rag/corpus/` as the default
  retrieval path.
- Optionally switches Stage 3 to Track A Vector RAG through
  `context-engineering/` when `context-engineering/vector_db/` has been built.
- Keeps the UI usable without API keys by falling back to deterministic or stub
  paths for intent, retrieval, content generation, and evaluation.
- Provides side pages for API/model settings and profile registry management.

## Demo Stages

1. Intent classification
   The user prompt is mapped to task codes, role, sector, topic keywords, and
   language. With an Intent API key it uses an LLM; otherwise it uses a
   deterministic keyword fallback.

2. Profile selection
   The classified intent filters workflow profiles and domain-expert profiles
   from `profiles/`.

3. Context retrieval
   The demo retrieves writing-guide rules and prior article examples. The
   default backend is PageIndex over cached local JSON trees. The UI can also
   switch to Vector RAG, which uses the `context-engineering/` hybrid retrieval
   adapter when its local vector database exists.

4. Prompt refinement
   The app asks clarification questions, folds answers into a refined prompt,
   and checks whether the refined prompt changed the intent enough to re-run
   profile selection and retrieval.

5. Content generation
   The refined prompt, selected profiles, and retrieved snippets are used to
   draft final content with citations. Without a Content Generation API key,
   this stage returns a visible stub so the flow can still be tested.

6. Evaluation
   Generated content is checked against the KPI catalogue. Tier 1 deterministic
   checks always run, Tier 2 LLM judge checks run when configured, and Tier 3
   reports required dCLP/editorial signoff steps.

## Layout

```text
/
├── data/                  # shared writing guide, KPI workbook, EN/NL article corpus
├── rag/corpus/            # shared generated corpus markdown + PageIndex trees
└── archive/
    ├── frontend/              # Streamlit app and pages
    ├── backend/               # intent, profile selection, retrieval, refinement, generation, evaluation
    ├── profiles/              # workflow and domain-expert YAML profiles
    ├── rag/                   # PageIndex retrieval code, notebooks, build scripts
    ├── context-engineering/   # Track A Vector RAG implementation
    ├── document_processing/   # markdown/document ingestion helpers
    ├── task_definition/       # earlier task-definition prototype
    ├── docs/                  # project notes, reports, diagrams, test queries
    └── results/               # generated PageIndex outputs
```

The archived docs may refer to `data/` and `rag/corpus/`; those paths now mean
the shared repo-root directories, not folders inside `archive/`.

## Setup

```bash
cd archive
uv sync
```

Optional API keys can be supplied in `archive/.env` or entered on the Streamlit
Settings page:

```bash
OPENAI_API_KEY_INTENT=...
OPENAI_API_KEY_PAGEINDEX=...
OPENAI_API_KEY_VECTOR_RAG=...
OPENAI_API_KEY_CONTENT_GENERATION=...
OPENAI_API_KEY_EVALUATION=...
```

## Run The Demo

```bash
cd archive
uv run streamlit run frontend/app.py
```

Open the URL printed by Streamlit. Start on the main AURORA page, enter a
writing request, then use the stage buttons to move through the pipeline.

Useful side pages:

- `Settings`: set API keys, model names, strict evaluation mode, and stage tinting.
- `Profiles`: view, add, edit, or delete workflow and domain-expert profiles.

## Rebuild PageIndex Assets

The checked-in cached trees are enough for the default demo path. To rebuild:

```bash
cd archive
uv run python rag/scripts/build_corpus.py
uv run python rag/scripts/run_pageindex.py --pdf_path ../data/"Writing Guide 2026-V1.1.pdf"
uv run python rag/scripts/run_pageindex.py --md_path ../rag/corpus/corpus_en.md
cp results/corpus_en_structure.json ../rag/corpus/corpus_en_structure.json
uv run python rag/scripts/enrich_structure.py
```

Generated PageIndex outputs land in `archive/results/`; the canonical corpus
cache lives in the repo-root `rag/corpus/`.

## Build Vector RAG Assets

Vector RAG is optional. It needs extra dependencies from
`context-engineering/requirements.txt` and writes a local
`context-engineering/vector_db/` directory.

```bash
cd archive/context-engineering
pip install -r requirements.txt
python -m scripts.ingest_pdf
python -m scripts.chunk --method a9 --src gpt-5
python -m scripts.chunk --method a10 --src writing-guide
python -m scripts.embed --embedder e4 --chunker a9 --src gpt-5
python -m scripts.embed --embedder x4 --chunker a9 --src gpt-5
python -m scripts.embed --embedder x4 --chunker a10 --src writing-guide
```

After this, launch the Streamlit app from `archive/` and select `Vector RAG` at
the top of the main page.

## Test And Validate

```bash
cd archive
uv run python -m py_compile frontend/app.py backend/retrieval/context_engineering_provider.py
uv run pytest
```

The Vector RAG end-to-end retrieval test is skipped automatically until both
`context-engineering/vector_db/` and the optional Track A dependencies exist.

## Notebooks

```bash
cd archive
uv run jupyter lab rag/notebooks/
```

`rag/notebooks/aurora_demo.ipynb` is the notebook-style end-to-end demo.
`rag/notebooks/crash_course.ipynb` is the PageIndex tutorial.
