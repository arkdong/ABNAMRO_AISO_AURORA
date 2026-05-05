# Status

Living document — what is done, in progress, and planned next.
Update this file at the end of each working session.

## Done

- **2026-04-29 — Project bootstrap.** Created folder layout, stub scripts for
  every pipeline stage, `requirements.txt`, `.env.example`, `.gitignore`,
  `README.md`. 
- **2026-04-29 — Advanced strategy docs.** Wrote [`advanced-chunking.md`](advanced-chunking.md)
  (A1–A8) and [`advanced-embedding.md`](advanced-embedding.md) (X1–X11) covering
  contextual retrieval, late chunking, RAPTOR, small-to-big, HyDE-indexing,
  reranking, hybrid BM25+dense, ColBERT, BGE-M3 multi-output, etc.
- **2026-04-30 — Scraper working.** `scripts/scrape.py` discovers article
  URLs via the public year-by-year sitemap (`/insights/sitemap/<topic>/<year>.html`)
  — found 491 candidate articles. Static HTML, no Playwright needed.
  Extracts title (og:title), URL (og:url), date (DD/MM/YYYY in body header),
  sector + topic (JSON-LD BreadcrumbList positions 3–4), body (`<article>`
  tag, scripts/styles stripped). Deterministic shuffle for sector diversity.
  Ran `--limit 20`, got a clean spread across agrarisch, bouw, industrie,
  food, leisure, technologie, and others.
- **2026-04-30 — Translator working.** `scripts/translate.py` uses
  `deep-translator` (Google Translate) with paragraph-batched chunking
  (4500-char limit) so >5000-char articles translate cleanly. Preserves
  original NL text in `body_original` / `title_original` for later
  research question Q1 (translate-vs-raw). Translated all 20 scraped
  articles, including a 24,921-char retail article in multiple batches —
  output reads naturally, technical terms preserved.

## In progress

_(nothing yet)_

## Next

1. **Writing Guide ingestion** — extract text from `data/writing_guide.pdf`,
   store as a parallel "doc_type=writing_guide" source for chunking/embedding.
2. **Chunking experiments** — implement strategies in
   [`experiments.md`](experiments.md) (C1–C10) over `data/translated/` and
   `data/writing_guide.pdf`. Pick a baseline (C1 fixed-size, C4 paragraph),
   add A5 small-to-big and A1 contextual retrieval as cheap wins.
3. **Embedding experiments** — index each chunking output, run the eval
   queries from `tests/test_retrieval.py`.
4. **Eval set** — expand `tests/test_retrieval.py` with 10–15 hand-labeled
   queries. Without this, the matrix in `experiments.md` is hand-waving.
5. **Intent classifier** — Claude or OpenAI call returning the JSON schema
   defined in `scripts/classify.py`.
6. **Retrieval** — already stubbed; wire to the chosen chunker + embedder.

## Backlog / out of scope for MVP

- Other channels (chatbot, app, push)
- Ruby channel guidelines
- Embedding fine-tuning
- Azure deployment
