# rag

Local PageIndex RAG. No hosted PageIndex API — everything runs against the vendored library in `pageindex/` using OpenAI (or any LiteLLM-supported provider) for the LLM calls.

## Subdirectories

- `pageindex/` — vendored upstream library (`page_index_main`, `md_to_tree`, `PageIndexClient`).
- `scripts/build_corpus.py` — concatenates EN articles in repo-root `data/article/en/` into repo-root `rag/corpus/corpus_en.md`.
- `scripts/run_pageindex.py` — CLI: turn a PDF or markdown file into a tree-structured JSON.
- `notebooks/` — `aurora_demo.ipynb` (end-to-end demo), `crash_course.ipynb` (PageIndex tutorial).
- `examples/` — selected upstream reference examples.
- `legacy/` — pre-reorg scratch material.

The canonical corpus artifacts live outside the archive at repo-root
`rag/corpus/`: `corpus_en.md`, `corpus_en_structure.json`,
`corpus_en_manifest.json`, and `writing_guide_tree.json`.

## Quickstart

```python
from rag.pageindex import PageIndexClient

client = PageIndexClient()  # uses OPENAI_API_KEY from env
result = client.submit_document("../data/Writing Guide 2026-V1.1.pdf")
tree = client.get_tree(result["doc_id"])
```
