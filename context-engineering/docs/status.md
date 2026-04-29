# Status

Living document — what is done, in progress, and planned next.
Update this file at the end of each working session.

## Done

- **2026-04-29 — Project bootstrap.** Created folder layout, stub scripts for
  every pipeline stage, `requirements.txt`, `.env.example`, `.gitignore`,
  `README.md`. Copied Writing Guide PDF into `data/writing_guide.pdf`.

## In progress

_(nothing yet)_

## Next

1. **Scraper** — inspect `abnamro.nl/zakelijk/insights` (static vs JS), build
   article URL discovery, extract title/date/sector/body, save to `data/raw/`.
2. **Translator** — wire `deep-translator` over `data/raw/` into `data/translated/`.
3. **Chunking experiments** — implement the strategies listed in
   [`experiments.md`](experiments.md), save outputs side-by-side.
4. **Embedding experiments** — index each chunking output with multiple
   embedding models, run the eval queries from [`experiments.md`](experiments.md).
5. **Intent classifier** — Claude or OpenAI call returning the JSON schema
   defined in `scripts/classify.py`.
6. **Retrieval** — already stubbed; wire to the chosen chunker + embedder.

## Backlog / out of scope for MVP

- Other channels (chatbot, app, push)
- Ruby channel guidelines
- Embedding fine-tuning
- Azure deployment
