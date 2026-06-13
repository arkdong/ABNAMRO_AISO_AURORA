# document_processing

Web scraper for ABN AMRO Insights articles. Captures each article into
Markdown with YAML frontmatter, matching the format of `data/article/nl/`.
Frontmatter is produced by an Obsidian Web Clipper-style template engine, so
swapping selectors as the site evolves only requires editing the template
strings — not the parser.

## Layout

```
document_processing/
├── cli.py                       # python -m document_processing ...
├── scrapers/
│   ├── abnamro_scraper.py       # fetch + parse one article URL
│   ├── feeds.py                 # crawl actueel.html for article URLs
│   ├── selectors.py             # CSS selector constants (single point of change)
│   └── template_engine.py       # Obsidian Web Clipper template evaluator
├── examples/
│   └── obsidian_template.json   # reference template (mirrored in PROPERTIES)
└── tests/fixtures/              # drop saved HTML pages here for offline tests
```

## Install

The required dependencies are already used elsewhere in the repo
(`context-engineering/requirements.txt`). From the project root:

```bash
uv pip install requests beautifulsoup4 markdownify pyyaml
```

## Usage

Single article:

```bash
python -m document_processing scrape-url \
  "https://www.abnamro.nl/nl/zakelijk/insights/sectoren-en-trends/zakelijke-dienstverlening/ai-succesvol-inzetten-vraagt-meer-dan-technologie.html"
```

Crawl the actueel.html feed and scrape everything it links to:

```bash
python -m document_processing scrape-feed
python -m document_processing scrape-feed --limit 10              # cap for testing
python -m document_processing scrape-feed --out-dir ../data/article/nl --force
```

Output filename is the article title (e.g. `AI succesvol inzetten vraagt meer dan technologie.md`).

## Output format

Frontmatter mirrors `data/article/nl/`:

```yaml
---
title: "..."
source: "https://..."
lang: "nl"
published: 2026-01-19
author:
  - "[[First Last]]"
description: "..."
tag:
  - "[[Analyse]]"
  - "[[Technologie, Media & Telecom]]"
---
<article body in Markdown — images and iframes preserved>
```

## Translation (NL → EN)

Set the dedicated translation key in a `.env` at the project root so its OpenAI
spend stays separate from other AURORA components:

```bash
OPENAI_API_KEY_TRANSLATION=sk-...   # preferred — translation cost lives here
OPENAI_API_KEY=sk-...               # fallback for the rest of the project
```

Translate one article (smoke test):

```bash
python -m document_processing translate \
  --file "../data/article/nl/<some article>.md" \
  --dst ../data/article/en
```

Translate everything in the shared repo-root `data/article/nl/`:

```bash
python -m document_processing translate --src ../data/article/nl --dst ../data/article/en
python -m document_processing translate --model gpt-5-mini   # cheaper option
```

The CLI prints token usage per article and an estimated total cost at the end.
Output filename matches the source NL filename, which acts as a stable
parallel-corpus key across languages.

## Notes / known limitations

- `crawl_actueel` parses the **static** HTML of `actueel.html`. If ABN AMRO
  loads tiles via JavaScript, this will return few/no URLs. Fallback options:
  walk the year-by-year sitemap (proven approach in
  `context-engineering/scripts/scrape.py`) or add Playwright.
- The body uses `markdownify` with images and iframes preserved (matching the
  Obsidian Web Clipper output convention).
- No translation. Translation is intentionally out of scope for this module.
