# PageIndex scale-up plan (10 → 59 articles)

Tracks the work to rebuild the retrieval index over the full English article
set in `data/article/en/` and make ranking scale past ~10 documents.

Owner: Adam. Last updated: 2026-05-12.

## Context

- `data/article/en/` now contains 59 translated articles (was ~10).
- The backend retrieval layer (`backend/retrieval/pageindex_provider.py`) loads a
  cached PageIndex tree from `rag/corpus/corpus_en_structure.json`. That JSON
  was built from the old 10-article corpus and is stale.
- The build pipeline already exists:
  - `rag/scripts/build_corpus.py` — concatenates articles into `rag/corpus/corpus_en.md`.
  - `rag/scripts/run_pageindex.py` — runs `md_to_tree` and emits a structure JSON.
- Frontmatter (title, description, tag, published, source) is currently
  dropped by `build_corpus.py` — only `title` is preserved.

## Status tracker

| Phase | Status | Notes |
|---|---|---|
| 1. Regenerate index over all 59 articles | DONE (2026-05-12) | `corpus_en_structure.json` rebuilt: 59 top-level / 345 total nodes. |
| 2. Preserve frontmatter as article-level metadata | DONE (2026-05-12) | Manifest emitted by `build_corpus.py`; `enrich_structure.py` joins it back into the tree. All 59/59 nodes enriched. |
| 3. Two-stage LLM ranking | DONE (2026-05-12) | `_llm_pick` now does stage-1 article shortlist (title+desc+tags only) then stage-2 section rank on the shortlisted subtrees. Verified live with gpt-4o on the demo query. |
| 4. Tag-aware pre-filter in `_CORPUS_ROUTING` | DONE (2026-05-12) | `_filter_by_tags` narrows top-level nodes by `topic_keywords` overlap with article tags; only narrows when match count is `>= 1` and strictly less than the input. Sector is informational, not a narrowing term. Applies to both LLM and deterministic paths. |

### What changed during phase 1+2 execution

- `rag/scripts/build_corpus.py` now (a) strips any inline body H1 that would duplicate the script-prepended title (22 of 59 articles had this), and (b) emits `rag/corpus/corpus_en_manifest.json` from frontmatter (title, description, tag, published, source, author).
- `rag/scripts/run_pageindex.py` had two bugs that blocked the markdown path:
  - Stray `from pageindex.utils import ConfigLoader` import — removed.
  - `user_opt` dict passed `None` values that overrode YAML defaults — fixed by dropping `None` entries before merge (matches the PDF branch).
- New: `rag/scripts/enrich_structure.py` — joins manifest to top-level nodes in `corpus_en_structure.json` by normalised title; overwrites `prefix_summary` with frontmatter `description` and adds `tags`, `published`, `source`, `slug`.
- Output of `run_pageindex.py` still lands in cwd-relative `./results/` — copy to `rag/corpus/corpus_en_structure.json` after each run. (Worth fixing to write into the canonical path directly, but out of scope for phase 2.)

### What changed during phase 3+4 execution

- `backend/retrieval/pageindex_provider.py`
  - Added `_filter_by_tags` / `_query_terms` / `_tag_matches` (phase 4). Narrows top-level nodes by `topic_keywords ↔ tags` substring overlap; only narrows when at least one match and the matched set is strictly smaller than the input. Sector is intentionally excluded from narrowing — too coarse in a single-sector corpus.
  - Rewrote `_llm_pick` as a two-stage ranker (phase 3): stage 1 sees only article titles+descriptions+tags (compact prompt); stage 2 ranks sections within the shortlisted articles.
  - Bug fixes on the critical path:
    - `_tokens` now drops English+Dutch stopwords (otherwise short writing-guide section titles dominate keyword overlap at 59-article scale).
    - `_query_token_set` tokenizes `topic_keywords` separately at `min_len=2` so acronyms like `AI` survive.
- `rag/pageindex/config.yaml`: `if_add_node_text` flipped to `"yes"`. Required by `pageindex_provider` — the provider expects every node to carry its text inline. Without this, the regenerated structure has no `text` and every node fails the snippet builder's check.

### How to re-run end-to-end

```sh
set -a && source .env && set +a
export OPENAI_API_KEY="${OPENAI_API_KEY:-$OPENAI_API_KEY_TRANSLATION}"
python rag/scripts/build_corpus.py
python rag/scripts/run_pageindex.py --md_path rag/corpus/corpus_en.md
cp results/corpus_en_structure.json rag/corpus/corpus_en_structure.json
python rag/scripts/enrich_structure.py
```

## Phase 1 — Regenerate the index

Steps:

1. Run `python rag/scripts/build_corpus.py` → writes `rag/corpus/corpus_en.md`
   with all 59 articles as `# Title` / `## Section` blocks.
2. Run `python rag/scripts/run_pageindex.py --md_path rag/corpus/corpus_en.md`
   with `if_add_node_summary=yes` (config default). Outputs to
   `rag/results/corpus_en_structure.json`.
3. Move/copy that file to `rag/corpus/corpus_en_structure.json`. Backend
   (`backend/retrieval/corpus_loader.py:20`) picks it up automatically.

Risks: LLM cost (~300 summary calls). No code change, fully reversible by
restoring the old JSON from git.

## Phase 2 — Preserve frontmatter as article-level metadata

The frontmatter holds the cleanest article-level summary (`description`) and
the routing-friendly tags (`tag`). Plan:

1. **Edit `rag/scripts/build_corpus.py`** to additionally emit
   `rag/corpus/corpus_en_manifest.json` — a list of `{slug, title, published,
   description, tag, source}` records, one per article. Slug = file stem.
2. **Add `rag/scripts/enrich_structure.py`** — post-process pass that:
   - Loads `corpus_en_structure.json` and `corpus_en_manifest.json`.
   - Joins each top-level (`#`) node to a manifest entry by title.
   - Injects `description` into the top-level node's `prefix_summary` (better
     than what the LLM generates from the body alone).
   - Adds `tags`, `published`, `source` fields to the top-level node.
   - Writes the enriched tree back to `corpus_en_structure.json`.
3. **No backend change required** — `pageindex_provider.py` already reads
   `prefix_summary` and ignores unknown extra fields.

## Phase 3 — Two-stage LLM ranking (deferred)

`backend/retrieval/pageindex_provider.py:146` (`_llm_pick`) currently flattens
the whole tree into one prompt. With 59 articles × ~5 sections, this is ~300+
nodes per request — expensive and accuracy degrades.

Proposed change:
1. **Stage 1** — send only top-level nodes (article titles + descriptions +
   tags). Ask LLM to pick top ~5 articles.
2. **Stage 2** — send the full subtree of just those 5 articles. Pick best
   sections.

Fall back to single-stage on either failure.

## Phase 4 — Tag-aware routing (deferred)

`_CORPUS_ROUTING` in `pageindex_provider.py:27` routes by task code only. Once
tags are on nodes (phase 2), add a cheap pre-filter that intersects
`RetrievalQuery.topic_keywords` with article `tags` before LLM ranking.
