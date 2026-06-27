# Dutch and Bilingual Support

This document records where Dutch support lives in AURORA, what changed to make
the system Dutch-capable, and how to rebuild the Dutch PageIndex and vector RAG
assets.

## Source Documents

Dutch source material is stored at the repository root:

| Source | Path | Purpose |
|---|---|---|
| Dutch Insights articles | `data/article/nl/*.md` | Approved Dutch article corpus for grounding, examples, and source-backed drafting |
| English Insights articles | `data/article/en/*.md` | Approved English article corpus |
| Dutch writing guide | `schrijfwijzer.pdf` | Dutch ABN AMRO Schrijfwijzer used as writing-reference corpus |
| English writing guide | `data/Writing Guide 2026-V1.1.pdf` and `data/writing_guide.md` | English writing-reference source material |
| Dutch Insights style guide | `Insights_Stijlgids_20250318.pdf` | Dutch Insights style rules for formats, SEO, BX, accessibility, visuals, charts, PDFs, and documents |
| English Insights style guide | `data/insights_stijlgids_en.md` | English translation of the Dutch Insights style guide for English retrieval |

The live tool server does not read these source documents directly at runtime.
It reads generated assets under `aurora_tool_server/assets/rag/`.

## Generated Runtime Assets

The Dutch implementation adds these runtime assets:

| Asset | Path | Used by |
|---|---|---|
| Dutch PageIndex corpus | `aurora_tool_server/assets/rag/corpus_nl_structure.json` | `pageindex` retrieval for Dutch article grounding |
| Dutch Schrijfwijzer tree | `aurora_tool_server/assets/rag/schrijfwijzer_tree.json` | `pageindex` retrieval for Dutch writing rules |
| Dutch Insights style-guide tree | `aurora_tool_server/assets/rag/insights_stijlgids_nl_tree.json` | `pageindex` retrieval for Dutch Insights format/style rules |
| English Insights style-guide tree | `aurora_tool_server/assets/rag/insights_stijlgids_en_tree.json` | `pageindex` retrieval for English Insights format/style rules |
| Dutch article vector chunks | `aurora_tool_server/assets/rag/vector_corpus_nl.jsonl` | `vector_rag` retrieval |
| Dutch Schrijfwijzer vector chunks | `aurora_tool_server/assets/rag/vector_schrijfwijzer.jsonl` | `vector_rag` retrieval |
| Dutch Insights style-guide vector chunks | `aurora_tool_server/assets/rag/vector_insights_stijlgids_nl.jsonl` | `vector_rag` retrieval |
| English Insights style-guide vector chunks | `aurora_tool_server/assets/rag/vector_insights_stijlgids_en.jsonl` | `vector_rag` retrieval |
| English article vector chunks | `aurora_tool_server/assets/rag/vector_corpus_en.jsonl` | `vector_rag` retrieval |
| English writing-guide vector chunks | `aurora_tool_server/assets/rag/vector_writing_guide.jsonl` | `vector_rag` retrieval |

The final submission no longer keeps root `rag/corpus/` build artifacts. The
builder writes directly to the live runtime asset directory,
`aurora_tool_server/assets/rag/`.

## Rebuilding Assets

Run the builder from the live server package:

```bash
cd aurora_tool_server
uv run python scripts/build_rag_assets.py
```

The builder:

1. Reads `data/article/nl/*.md`.
2. Builds `corpus_nl.md`, `corpus_nl_manifest.json`, and
   `corpus_nl_structure.json`.
3. Extracts text from `schrijfwijzer.pdf` with `pdftotext`.
4. Builds `schrijfwijzer_tree.json`.
5. Extracts `Insights_Stijlgids_20250318.pdf` and builds
   `insights_stijlgids_nl_tree.json`.
6. Reads `data/insights_stijlgids_en.md` and builds
   `insights_stijlgids_en_tree.json`.
7. Builds local sparse-vector JSONL files for `vector_rag`.
8. Writes runtime assets into `aurora_tool_server/assets/rag/`.

The current `vector_rag` implementation uses local sparse term vectors stored
in JSONL. These files are embedding-ready, but they do not require a network
call or an external embedding service to work.

## Runtime Language Contract

The request schema now supports an explicit output-language override:

```python
StageOptions(output_language="nl")
StageOptions(output_language="en")
StageOptions(output_language="both")
```

This is separate from the language detected from the prompt. For example:

- Dutch user prompt, Dutch output: `Schrijf een artikel over cyberveiligheid.`
- English user prompt, Dutch output: `Write about cybersecurity.`, with
  `output_language="nl"`.
- Bilingual output: any prompt with `output_language="both"`.

The field flows through:

1. intent classification
2. profile selection
3. retrieval routing
4. refinement
5. generation
6. evaluation

## Retrieval Routing

Language-aware retrieval is implemented in
`aurora_tool_server/aurora_tool_server/retrieval.py`.

Routing rules:

| Language | Article corpora | Writing-reference corpora |
|---|---|---|
| `en` | `corpus_en` | `writing_guide`, `insights_stijlgids_en` |
| `nl` | `corpus_nl` | `schrijfwijzer`, `insights_stijlgids_nl` |
| `both` | `corpus_nl`, `corpus_en` | `schrijfwijzer`, `insights_stijlgids_nl`, `writing_guide`, `insights_stijlgids_en` |

Task-aware routing still applies:

- Drafting and renewal use article corpora plus writing references.
- Search uses article corpora.
- Compliance uses writing references first, then article corpora.
- Translation can include both language corpora because source and target may differ.

Both supported retrieval backends now use the same language routing:

```python
retrieval_backend="pageindex"
retrieval_backend="vector_rag"
```

## Generation Rules

Generation is implemented in `aurora_tool_server/aurora_tool_server/generation.py`.

For Dutch output, the system prompt now requires:

- Dutch output.
- Nederlandse ABN AMRO Schrijfwijzer.
- Formal `u`-form.
- B1/plain language.
- Active sentences.
- Clear headings.
- ABN AMRO Insights tone.
- No English drift except proper nouns, cited source titles, or unavoidable
  technical terms.

For bilingual output, the model is instructed to return clearly labelled Dutch
and English Markdown sections. The Dutch section follows the Schrijfwijzer; the
English section uses British English.

The deterministic fallback also returns Dutch scaffold text when
`intent.language == "nl"`.

## Refinement Rules

Refinement is implemented in `aurora_tool_server/aurora_tool_server/refinement.py`.

When Dutch output is requested, deterministic clarification questions use Dutch
copy, for example:

- `Voor wie is de tekst bedoeld?`
- `Er zijn geen sterke bronfragmenten gevonden. Moet AURORA toch doorgaan?`
- `Moet AURORA de huidige scope en bronnen aanhouden?`

The LLM refinement prompt also tells the model to preserve the requested output
language and, for Dutch, explicitly include Schrijfwijzer constraints in the
final refined prompt.

## Intent Detection

Intent detection is implemented in `aurora_tool_server/aurora_tool_server/intent.py`.

The deterministic classifier recognizes Dutch task and language cues such as:

- `schrijf`
- `artikel`
- `vertaal`
- `controleer`
- `beoordeel`
- `zoek`
- `nederlands`
- `in het nederlands`
- `naar engels`

The LLM classifier still returns the same typed contract:

```json
{
  "language": "en | nl | both | null"
}
```

## Agent Behavior

The OpenAI Agents SDK wrapper lives in
`aurora_tool_server/frontend/agent_service.py`.

It now instructs the agent to mirror the user's language. If the user writes in
Dutch, or if AURORA returns `intent.language="nl"`, the agent should answer in
Dutch and preserve Dutch drafts rather than translating them back to English.

## Evaluation

The deterministic evaluation layer lives in
`aurora_tool_server/aurora_tool_server/evaluation/tier1_deterministic.py`.

Dutch support adds a `language` KPI check:

- If Dutch output is requested, the checker expects Dutch language markers.
- If English output is requested, the checker expects English language markers.
- If bilingual output is requested, either language can pass the deterministic
  marker check.

This catches the main failure mode where the retrieval is Dutch but the draft
comes back in English.

The existing passive-voice check already switches to Dutch passive markers when
`intent.language == "nl"`.

## Tests

Dutch regression tests live in:

```text
aurora_tool_server/tests/test_dutch_support.py
```

The tests cover:

- Dutch prompt language detection.
- `output_language="nl"` override.
- Dutch PageIndex routing to `corpus_nl` and `schrijfwijzer`.
- Dutch `vector_rag` routing to generated vector assets.
- Dutch deterministic generation fallback.
- Dutch LLM generation prompt rules.
- Dutch deterministic refinement questions.

Run all live server tests:

```bash
cd aurora_tool_server
uv run pytest
```

Last verified result after implementation:

```text
65 passed
```

## Operational Examples

Dutch prompt, Dutch output:

```json
{
  "user_prompt": "Schrijf een kort artikel over AI-agents en softwarebedrijven.",
  "options": {
    "retrieval_backend": "pageindex",
    "k": 5
  }
}
```

English prompt, Dutch output:

```json
{
  "user_prompt": "Write a short article about cybersecurity in the TMT sector.",
  "options": {
    "output_language": "nl",
    "retrieval_backend": "pageindex",
    "k": 5
  }
}
```

Bilingual output:

```json
{
  "user_prompt": "Explain how AI agents affect SaaS companies.",
  "options": {
    "output_language": "both",
    "retrieval_backend": "vector_rag",
    "k": 5
  }
}
```

Expected Dutch runtime routing:

```text
corpora_searched = ["corpus_nl", "schrijfwijzer"]
```

Expected bilingual runtime routing:

```text
corpora_searched = ["corpus_nl", "corpus_en", "schrijfwijzer", "writing_guide"]
```

## Important Notes

- Runtime code reads `aurora_tool_server/assets/rag/*`, not the source files in
  `data/`.
- Re-run `scripts/build_rag_assets.py` whenever Dutch articles or
  `schrijfwijzer.pdf` change.
- The current vector backend is local sparse-vector retrieval. Dense embedding
  support can be added later without changing the public `vector_rag` option.
- The API/MCP endpoints did not need Dutch-specific duplicates. Dutch and
  bilingual behavior is controlled through `language` and `output_language`.
