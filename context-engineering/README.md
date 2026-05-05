# AURORA MVP

Content generation assistant for ABN AMRO content writers. Retrieves writing
guidelines and generates compliant insight articles for `abnamro.nl/zakelijk/insights`.

## Pipeline

1. `scripts/scrape.py` — scrape Dutch insight articles → `data/raw/*.json`
2. `scripts/translate.py` — Dutch → English → `data/translated/*.json`
3. `scripts/chunk.py` — chunk articles + Writing Guide → `data/chunked/*.json`
4. `scripts/embed.py` — embed chunks into local Chroma at `vector_db/`
5. `scripts/classify.py` — extract topic/sector/intent from a user request
6. `scripts/retrieve.py` — dual retrieval (style refs + writing rules)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

If you use OpenAI embeddings or Anthropic for classification, copy `.env.example`
to `.env` and fill in the keys.

## Folder layout

```
context-engineering/
├── data/
│   ├── raw/          scraped articles (Dutch)
│   ├── translated/   English translations
│   ├── chunked/      chunked + metadata, ready to embed
│   └── writing_guide.pdf
├── docs/             status, experiments log, research questions
├── scripts/          pipeline scripts
├── vector_db/        Chroma persistent store
└── tests/
```

## Where to look

- [docs/status.md](docs/status.md) — what's done, in progress, next.
- [docs/experiments.md](docs/experiments.md) — chunking × embedding matrix and results.
- [docs/research-questions.md](docs/research-questions.md) — open questions.

## Scope

Single channel (`website`), single content type (`insight_article`).
Out of scope: chatbot/app/push channels, fine-tuning, Azure deploy.
