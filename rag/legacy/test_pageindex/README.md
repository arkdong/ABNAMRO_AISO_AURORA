# PageIndex local POC for ABN AMRO Insights

This starter kit is built for your current corpus:

- 1 normative guide: **Writing Guide 2026 Version 1.1**
- 5 approved example articles from the Insights site

The POC is intentionally small and opinionated:

1. **Use PageIndex to generate tree JSON locally** for the PDF guide and the 5 markdown articles.
2. **Select documents from a prompt** using a small registry and PageIndex-style document descriptions.
3. **Retrieve relevant sections** from the selected PageIndex trees.
4. **Assemble the return context** that you would pass to your article-writing or article-review LLM.

## Why this setup

For a corpus of only six documents, **description-based document selection** is the cleanest first step.
You do not need a vector DB or a full metadata-SQL layer yet.

In this corpus, the writing guide should be treated as **normative authority** and the five articles as
**approved examples**.

## Folder layout

- `corpus/` — the 6 source files
- `corpus_manifest.json` — lightweight registry
- `scripts/01_generate_pageindex_trees.sh` — batch commands for local PageIndex indexing
- `scripts/select_and_retrieve.py` — select docs, retrieve sections, build context
- `prompts/` — prompt templates mirroring the PageIndex doc-search and tree-search style
- `pageindex_results/` — empty placeholder; your actual `*_structure.json` files will usually come from `PageIndex/results/`

## Step 1 — clone PageIndex and generate the trees

Follow the PageIndex README for local use:

```bash
git clone https://github.com/VectifyAI/PageIndex.git
cd PageIndex
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade -r requirements.txt
```

Create a `.env` file in the PageIndex repo root:

```bash
OPENAI_API_KEY=your_key_here
```

Then run the batch script from inside the cloned `PageIndex` repo:

```bash
bash ../pageindex_local_poc/scripts/01_generate_pageindex_trees.sh
```

That script runs:

- `run_pageindex.py --pdf_path ...` for the writing guide
- `run_pageindex.py --md_path ...` for each article
- `--if-add-node-summary yes`
- `--if-add-node-text yes`
- `--if-add-doc-description yes`

The PageIndex CLI saves outputs in:

```bash
./results/*_structure.json
```

## Step 2 — run document selection and section retrieval

Still from anywhere on your machine:

```bash
python3 pageindex_local_poc/scripts/select_and_retrieve.py   --manifest pageindex_local_poc/corpus_manifest.json   --pageindex-results PageIndex/results   --query "Refine a draft on AI adoption in European tech companies for ABN AMRO Insights. Keep it concrete, warm and authoritative."   --print-prompts   --output-json pageindex_local_poc/output/refine_ai_context.json
```

## What the script returns

The script prints three things:

1. `SELECTED DOCUMENTS`
2. `SELECTED SECTIONS`
3. `ASSEMBLED CONTEXT`

The assembled context is the exact pack you would send into your downstream LLM.

It also optionally prints:

- the **document-selection prompt**
- the **section-retrieval prompt**

So you can show stakeholders the retrieval logic clearly.

## Selection policy used in this starter kit

- If the query looks editorial (`refine`, `review`, `draft`, `style`, `tone`, etc.), the script **always includes the Writing Guide**.
- It then scores the 5 approved articles by overlap with:
  - title
  - description
  - sector
  - audience
- For the current corpus, technology / TMT prompts are biased toward the four technology examples.

## Retrieval policy used in this starter kit

For each selected document, the script loads the PageIndex tree JSON and flattens its nodes.

Each node is scored using:

- title overlap with the user prompt
- summary overlap with the user prompt
- text overlap with the user prompt
- a small boost for style-guide nodes that mention structure, tone, short sentences, active voice, jargon, accuracy, and related terms

The script then keeps the top sections and emits:

- document title
- section path
- page or line reference
- node id
- short excerpt

## Good demo prompts

See `example_queries.txt`.

A good first live demo prompt is:

> Refine a draft on AI adoption in European tech companies for ABN AMRO Insights. Keep it concrete, warm and authoritative. Show which approved examples and writing-guide sections I should use.

## Expected behavior for this corpus

For an editorial AI/tech prompt, you should usually see the script choose:

- the writing guide
- `AI succesvol inzetten vraagt meer dan technologie`
- `De twee gezichten van Agentic AI: wapen en schild`
- `Digitale soevereiniteit biedt grote kansen voor Europa`
- sometimes also `Sterke groei keert terug in Technologie, Media en Telecom`

That is the right behavior for an initial POC:
guide first, then topical approved examples.

## Next step after this POC

Once this works, the natural v2 is:

- add a real metadata table
- promote selection from description-based to metadata-first
- add rejected drafts as anti-pattern documents
- wrap the flow in a tiny FastAPI or Streamlit UI
- store each run as an audit trail: prompt, selected docs, selected sections, final context pack