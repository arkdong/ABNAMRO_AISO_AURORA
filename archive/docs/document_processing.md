# Document Processing — Summary

A new top-level module `document_processing/` that scrapes ABN AMRO Insights
articles into Markdown and translates them NL → EN using OpenAI.

## Layout

```
document_processing/
├── cli.py                       # python -m document_processing <cmd>
├── markdown_io.py               # parse/write Obsidian-style frontmatter
├── scrapers/
│   ├── abnamro_scraper.py       # one-URL fetch + parse
│   ├── feeds.py                 # actueel.html + sitemap.xml discoverers
│   ├── selectors.py             # CSS selectors (single source of truth)
│   └── template_engine.py       # Obsidian Web Clipper template evaluator
├── translators/
│   └── openai_translator.py     # GPT-5 + Pydantic structured output
├── data/writing_guide_rules.md  # 3.2K-token distilled BrE rules
└── examples/obsidian_template.json
```

## Scraper

**What it does:** fetches an ABN AMRO article URL, evaluates an
Obsidian-Web-Clipper-style template (the JSON template the user supplied) against
the parsed HTML, and writes a Markdown file matching the existing
`data/article/nl/` format byte-for-byte.

**Output frontmatter:** `title`, `source`, `lang`, `published` (date),
`author[]`, `description`, `tag[]` (Obsidian wikilinks). Body uses
`markdownify` and preserves hero images, hyperlinks, headings, iframes.

**Discovery:** the public `actueel.html` page is JS-rendered and returns 0
links from a static fetch. Use `--sitemap-xml` instead — `…/zakelijk/sitemap.xml`
exposes ~3,000 URLs and is fully populated. Filter with `--path-contains` for
URL-path pre-filter and/or `--sector` for tag-based post-filter.

**Commands:**

```bash
# one URL
python -m document_processing scrape-url <article-url>

# all 59 TMT articles via sitemap
python -m document_processing scrape-feed \
  --sitemap-xml \
  --path-contains "/sectoren-en-trends/technologie/" \
  --sector "Technologie, Media & Telecom"
```

**Filename convention:** the article title (`{Title}.md`), not a slug —
matches the existing parallel-corpus convention.

## Translator

**What it does:** reads a NL `.md` file, sends `(title, description, body)`
to GPT-5 with a stable system prompt loaded from `data/writing_guide_rules.md`,
gets back a Pydantic-parsed `(title_en, description_en, body_en)` in one
round-trip, and writes the EN file with the **same NL filename** (parallel-corpus
alignment key).

**Frontmatter mapping:**

| Field | NL → EN |
|---|---|
| `title`, `description` | translated |
| `lang` | `"nl"` → `"en"` |
| `source`, `published`, `author`, `tag` | unchanged |
| `review_needed: true` + `review_issues` | added if structure check fails |

**Structure check:** before saving, counts of images / iframes / headings /
hyperlinks must match the source within tolerance, and length ratio must be
0.5×–1.6×. Failures don't block the save but flag the file for review.

**Writing-guide injection:** the 48-page PDF was distilled by hand into
`data/writing_guide_rules.md` (~3.2K tokens) covering BrE spelling/vocab,
plain-B1 tone, no greenwashing, gender-neutral language, numbers/dates/
currencies/capitals, punctuation, brand rules, and Markdown preservation.
Stable across calls so OpenAI prompt caching activates after the first
request.

**Cost split:** translator looks up `OPENAI_API_KEY_TRANSLATION` first, falls
back to `OPENAI_API_KEY`. Use a dedicated key labelled "AURORA-translation"
in your OpenAI dashboard to bill translation runs as a separate line item.

**Commands:**

```bash
# smoke-test on one file
python -m document_processing translate --file "../data/article/nl/<file>.md"

# all 59 articles
python -m document_processing translate --src ../data/article/nl --dst ../data/article/en

# cheaper: ~5× less per article
python -m document_processing translate --model gpt-5-mini
```

**GPT-5 gotcha:** `reasoning_effort="minimal"` is the default — translation
doesn't need thinking tokens and the chat-completions endpoint otherwise
appears to hang while the model reasons silently. Combined with a 180-s
client timeout and a `[call ]` heartbeat line so any future stalls are
visible immediately.

## Cost estimate

Measured on real article tokens (mean 2,296 / median 1,695 / max 5,439 input
tokens). For all 59 TMT articles:

| Model | Input | Output | Total |
|---|---:|---:|---:|
| **gpt-5** | $0.18 | $1.35 | **~$1.54** |
| gpt-5-mini | $0.04 | $0.27 | ~$0.31 |
| gpt-5-nano | $0.01 | $0.05 | ~$0.06 |

Prompt caching shaves ~5–10% off the input portion after the first call.
Pricing reflects OpenAI's Aug 2025 launch list — verify on
`platform.openai.com/pricing` before each run.

## Status

| | |
|---|---|
| Scraper | ✅ working — 59 TMT articles successfully scraped to `data/article/nl/` |
| Translator | ✅ built and wired; offline parse/render verified end-to-end |
| API smoke test | ⏸ pending user's `OPENAI_API_KEY_TRANSLATION` |
| Full batch | ⏸ pending smoke-test approval |

## Files added (this session)

- `document_processing/` — entire new module (~13 files, ~900 LOC)
- `data/article/nl/` — populated with 59 TMT-tagged articles
- `docs/document_processing.md` — this summary
