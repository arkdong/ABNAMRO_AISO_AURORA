# AURORA pipeline — current state

A snapshot of how the editorial co-pilot pipeline works after the PageIndex
scale-up and the post-RAG prompt refinement work. Treat this as the single
read-cold doc for anyone (including future-Claude) joining the project.

Last updated: 2026-05-15.

## Pipeline at a glance

```
  user prompt
      │
      ▼
┌────────────────────┐
│ 1. Intent           │  task_codes, sector, topic_keywords, language
│    classification   │  (LLM, deterministic fallback)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 2. Profile          │  Workflow profile + Domain experts
│    selection        │  (rule-based filter on task_codes / sector / kw)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 3. PageIndex RAG    │  k snippets ranked from the corpus tree
│    retrieval        │  (tag pre-filter → two-stage LLM, deterministic fallback)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 4. Prompt           │  Iterative Q+A — clarify the prompt grounded in
│    refinement       │  the snippets + profiles already on screen
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 5. Conditional      │  Re-classify on refined prompt; if pivot,
│    re-run           │  re-do steps 2 + 3 with new intent
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 6. Content          │  Structured-output LLM call over refined prompt +
│    generation       │  snippets + profile bundle (stub fallback)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 7. Evaluation       │  3-tier KPI scoring against ABN AMRO Content KPI
│                     │  catalogue: Tier 1 deterministic, Tier 2 LLM-judge,
│                     │  Tier 3 dCLP signoff flags. Auto-runs post-generation.
└────────────────────┘
```

Each stage produces a Pydantic-shaped result that the next stage consumes.
The same provider+model env (`intent_model`) feeds intent/refinement; PageIndex
retrieval uses its own dedicated key (`OPENAI_API_KEY_PAGEINDEX`); content
generation has its own (`OPENAI_API_KEY_CONTENT_GENERATION`); evaluation has
its own (`OPENAI_API_KEY_EVALUATION`).

### Stage 6/7 deep dives
- Content generation — `backend/content_generation/`, summary below.
- Evaluation — see `docs/evaluation_stage.md` (with workbook background in
  `docs/kpi_workbook_analysis.md`).

---

## How PageIndex RAG retrieval is done

### The index

- Source: 59 English-translated TMT articles in `data/article/en/`.
  `data/` is the shared repo-root data directory.
- Build pipeline (one-time, repeatable):
  1. `rag/scripts/build_corpus.py` concatenates all 59 articles into
     `rag/corpus/corpus_en.md`, stripping YAML frontmatter, image/iframe
     lines, byline, boilerplate CTAs, and any duplicated body H1. It also
     emits `rag/corpus/corpus_en_manifest.json` — one record per article
     with title, description, tag, published, source, author.
  2. `rag/scripts/run_pageindex.py --md_path ../rag/corpus/corpus_en.md` runs
     PageIndex's markdown tree builder. Each `#` heading becomes a top-level
     node (an article), each `##` becomes a child (a section). LLM
     summaries are generated per section. Output:
     `rag/corpus/corpus_en_structure.json` (59 top-level / 345 total nodes,
     all carrying inline `text`).
  3. `rag/scripts/enrich_structure.py` joins the manifest back into the
     tree by normalised title, overwriting each article's `prefix_summary`
     with the frontmatter `description` (better than the LLM-from-body
     summary), and adding `tags`, `published`, `source`, `slug` fields.

### The retrieval call

`PageIndexProvider.retrieve(query)` runs three layers, each cheaper-or-equal
than the next:

1. **Corpus routing** (`_route_corpora`).
   The intent's `task_codes` dictate which corpora to search:
   `T1_DRAFT` → `corpus_en + writing_guide`; `T1_SEARCH` → `corpus_en`;
   `T2_COMPLIANCE` → `writing_guide + corpus_en`; etc.

2. **Tag pre-filter** (`_filter_by_tags`).
   `topic_keywords` from the intent are substring-matched against each
   article's `tags`. If at least one article matches and the matched set is
   strictly smaller than all articles, narrow the candidate set. This is
   typically 59 → ~5–32 candidates depending on specificity. Sector is
   excluded from narrowing — the corpus is single-sector (TMT), so sector
   would match every article and starve the filter.

3. **Ranking** — two paths:
   - **LLM (two-stage), when an API key is configured.**
     - *Stage 1:* compact prompt of only top-level nodes (article titles +
       descriptions + tags). LLM picks the top ~5 articles.
     - *Stage 2:* full subtree of those 5 articles only. LLM ranks the best
       sections. The prompt stays small even at 59 articles.
   - **Deterministic fallback.**
     - Token-overlap scoring over title (3×) and summary (1×) tokens.
     - `_tokens` skips English+Dutch stopwords and short non-acronym tokens.
       `topic_keywords` are tokenized separately at `min_len=2` so `AI`
       survives.

The provider returns up to `query.k` `Snippet` objects, each with
`source_doc`, `node_id`, `title`, full `content`, score, and a one-line
human-readable `reason`.

### Why this layered design

- Tag pre-filter is cheap and high-precision when topic keywords match
  article tags. It dramatically shrinks the prompt the LLM sees.
- Two-stage LLM is necessary at scale: a single-stage prompt with 59
  articles × ~5 sections each gets noisy and expensive. Splitting into
  article-shortlist → section-pick keeps each prompt tight.
- The deterministic path is the always-on safety net — no API key, no LLM
  outage matters; the system still returns sensible results.

---

## How prompt improvement works (stage 4)

After retrieval, the frontend seeds a `kind: "refinement"` message. The
loop runs as a chat dialog:

1. **Generator call** (`backend/prompt_refinement/generator.py`).
   The LLM receives: original prompt, current refined prompt, intent,
   profile bundle summary, top retrieved snippets, prior turns. It returns
   a structured `GeneratorOutput`:
   - `questions: list[QuestionWithChoices]` — 1–3 sharp clarifying
     questions, each with 2–4 click-to-pick suggested answers (or empty
     `choices` for open-ended).
   - `proposed_prompt: str | None` — optional model-rewritten prompt.
   - `done: bool` — model can signal the prompt is already specific enough.
   - `reasoning: str` — single short sentence, logged only.

2. **Per-question rendering.**
   Each question is its own assistant chat bubble. The user can click any
   suggested choice (one tap → answer recorded) or type a custom reply.
   Their answer renders as a user bubble. Each Q+A pair appends to a
   `Clarifications:` block under the original prompt via
   `service.append_qa(...)`.

3. **Auto-advance.**
   When the user has answered every question in the current batch, the
   generator is called again with the updated state for the next batch.
   The loop caps at 5 clarifications.

4. **Footer (always visible).**
   A bottom panel shows the live refined prompt and two buttons:
   **Use this prompt** (lock in) and **Skip refinement** (bail to the
   current snippets unchanged).

When the deterministic fallback is in effect (no LLM key), a stub
generator produces heuristic questions with canned choices so the UI still
works end-to-end.

---

## Why refinement runs **after** RAG, not before

The natural urge is to refine first ("clean the question, then look") — we
chose the opposite. Reasoning:

- **Grounded > blind questions.** When refinement runs after retrieval,
  the LLM can reference actual articles ("We have *The two faces of
  Agentic AI* and *Cybersecurity in the TMT sector* — want to prioritise
  either?"). Without snippets, the LLM has to guess what's worth
  clarifying based on the prompt text alone, which produces vague,
  user-fatigue questions like "what's your timeframe?" that may or may not
  even be answerable from the corpus.
- **Profile selection already provides some shape.** By the time we hit
  refinement, the user has seen which workflow profile and which domain
  experts were activated. Those tell them *how* the task will be tackled.
  Snippets tell them *what's actually in the corpus*. Both inform smarter
  questions.
- **Most users under-specify.** They write a one-line prompt because they
  don't know what's possible. Showing them snippets first reveals the
  surface area — "oh, you have an article on EU cyber law? then I want
  that angle" — and turns the refinement into a *discovery* step, not a
  *guessing* step.
- **Cheap initial pass keeps the cost in check.** The first retrieval can
  default to deterministic (or wide-k LLM) — that's a few cents per query
  at most. The LLM ranker fires on the *refined* prompt, where the
  precision matters most.

### Tradeoffs we accepted

- **Wasted compute on pivots.** If a user comes in with "do something
  about cyber", retrieval will pull cyber snippets, then in refinement the
  user pivots to "actually I want 5G translation". The first retrieval is
  partially wasted. Mitigated by:
  - the deterministic-first default for stage 3 (~0 cost), and
  - the `needs_re_retrieval` heuristic that *only* re-runs profile
    selection + retrieval when the refined intent has actually shifted.
- **One extra round of latency.** Refinement is a blocking dialog after
  snippets render. Mitigated by allowing **Skip refinement** at every
  step, so users who already know what they want can bypass the loop.
- **Refinement is anchored to the initial retrieval.** If the initial
  retrieval is bad (rare, but possible if intent classification was way
  off), the refinement questions will be anchored to bad snippets. The
  pivot path is the escape hatch — explicitly designed so users can write
  a fundamentally different refined prompt and the system catches up.

### Why **not** put refinement **before** RAG

A pre-RAG grilling loop has three problems we couldn't solve cheaply:

1. **The model is asking blind.** Without snippets, the LLM has no
   evidence of what's worth clarifying. It falls back to generic
   questions ("Who's the audience?", "What length?") that the user may
   answer in a way the corpus can't satisfy. The user wastes turns
   specifying constraints we can't honour.
2. **The user is also asking blind.** They came in with a one-line prompt
   because they don't know what's available. Asking them to be more
   specific before they see anything is asking them to specify in the
   dark.
3. **Refinement isn't free.** Each turn is an LLM call. Doing N rounds
   *before* showing the user anything useful is friction with no visible
   payoff — they sit through a grill before they get a snippet to react to.

The post-RAG version trades a small extra retrieval cost for a much
sharper refinement loop and a faster perceived response. The first thing
the user sees is *snippets they can react to*; questions then build on
that shared context.

---

## How intent pivot is checked

`backend/prompt_refinement/service.needs_re_retrieval(orig_intent, refined_intent) -> bool`

After the user clicks **Use this prompt**, the frontend re-runs
`classify_full` on the refined text and compares the new `IntentResult` to
the original. It returns `True` (i.e., re-run profile selection + retrieval)
when **any** of the following hold:

| Check | Trigger |
|---|---|
| Task codes differ | `set(original.task_codes) != set(refined.task_codes)` |
| Sector differs | `(original.sector or "") != (refined.sector or "")` |
| Topic keywords drift | Jaccard of `topic_keywords` sets < 0.5 |

The Jaccard threshold of 0.5 was chosen empirically:

- Adding one keyword to a 2-keyword set → Jaccard 0.67 → not a pivot (the
  user *narrowed*, didn't pivot).
- Replacing all keywords (cyber → 5G) → Jaccard 0 → pivot.
- One-for-one swap in a 4-keyword set (3 retained, 1 changed) → Jaccard
  0.6 → not a pivot.

Sector and task code differences are treated as hard pivots — they change
which workflows even apply, so we always re-run.

When a pivot is detected, the frontend appends three new assistant
messages (intent, profiles, retrieval) below the refinement message, each
populated with the post-pivot results. No history is destroyed — the
original chain stays visible above for context.

When **no** pivot is detected, the refinement is still recorded (so
downstream generation sees the user's clarifications), but the existing
snippets are reused. Cheap, predictable.

---

## How content evaluation works (stage 7)

After Stage 6 generates the markdown body, the evaluator scores it against
ABN AMRO's Content KPI catalogue (134 KPIs, 7 categories, 22 clusters —
sourced from `data/Content KPI inventory_AISO.xlsx`). Three tiers:

1. **Tier 1 — deterministic** (always runs). 11 checks: sentence/paragraph
   length, passive voice, bullet-list presence, alt-tag coverage, reading
   level (Flesch → CEFR), H1 sanity, keyword-in-H1, source-citation
   presence (Tracability — Blocking), out-of-range citation indices
   (Factuality floor — Blocking), exclusion-tag check for GenAI sources
   (Approved-source — Blocking).
2. **Tier 2 — LLM judges** (parallel, configurable model). 12 rubrics each
   constrained to a workbook indicator-enum scale via Pydantic
   `response_format`: factuality, truthfullness, relevancy, privacy,
   groundedness, completeness, comprehensiveness, clarity, plus 4 GenAI
   search-quality-rater rubrics (uniqueness, expertise, no-paraphrase,
   no-filler).
3. **Tier 3 — dCLP signoff** (declared, not executed). 4 KPIs the workflow
   system must clear: human-substance check, legal/compliancy check,
   second-content-specialist check, evaluation-recency.

Service flow: load catalogue → Tier 1 → **short-circuit** if any
Mandatory + Blocking KPI failed (don't burn LLM spend) → otherwise run
Tier 2 in parallel via thread pool → Tier 3 pending entries → aggregate
`maturity_by_category` (low/medium/high per top-level category) +
`failed_blocking` (any Blocking KPI that didn't pass across all tiers).
Returns an `EvaluationResult` Pydantic envelope; never raises.

Stub path (no eval key): every Tier 2 KPI is recorded as `not_evaluated`
and (in lenient mode) passes; `strict_mode=True` flips that to failing.

The catalogue itself is built once via
`python rag/scripts/build_kpi_catalogue.py` (idempotent xlsx → JSON), and
loaded via an `lru_cache(maxsize=1)` typed accessor at runtime. openpyxl
is a build-time-only dep.

### Why three tiers

The workbook's 10 Blocking KPIs split cleanly: 5 are deterministic-checkable
(metadata presence + citation sanity), 5 are inherently judgemental
(factuality, truthfullness, relevancy, privacy, lawfullness audit).
Add the 4 dCLP signoff steps and you have three distinct compute models —
hence three tiers, not a single graph of checks.

### What it lands in the UI

Auto-runs at the end of Stage 6 and appends a cyan-tinted "Stage 7"
bubble: pass/fail verdict, per-category maturity dots, dCLP signoff queue,
collapsible per-KPI breakdown grouped by tier (failures floated above
passes within each weight band).

---

## Key files

| Area | File |
|---|---|
| Corpus build | `rag/scripts/build_corpus.py`, `rag/scripts/run_pageindex.py`, `rag/scripts/enrich_structure.py` |
| KPI catalogue build | `rag/scripts/build_kpi_catalogue.py` |
| Cached index | `rag/corpus/corpus_en_structure.json`, `rag/corpus/corpus_en_manifest.json`, `rag/corpus/corpus_en.md` |
| Cached KPI catalogue | `backend/evaluation/data/kpi_catalogue.json` |
| RAG provider | `backend/retrieval/pageindex_provider.py`, `backend/retrieval/corpus_loader.py`, `backend/retrieval/types.py` |
| Refinement | `backend/prompt_refinement/{types,generator,service}.py` |
| Content generation | `backend/content_generation/{types,prompt,service}.py` |
| Evaluation | `backend/evaluation/{types,indicators,catalogue,tier1_deterministic,tier2_judges,tier3_human_loop,prompt,service}.py` |
| Frontend | `frontend/app.py` (intent → profiles → retrieval → refinement → content → evaluation; replay loop in `messages`) |
| Plans / status | `docs/pageindex_scale_up_plan.md`, `docs/prompt_refinement_plan.md`, `docs/evaluation_stage.md`, `docs/kpi_workbook_analysis.md` |

## Environment

- `OPENAI_API_KEY_PAGEINDEX` — drives PageIndex tree generation (via
  `rag/pageindex/utils.py` + `rag/pageindex/client.py`) and the two-stage
  retrieval LLM ranker (`PageIndexProvider`). No other code path consumes
  this key.
- `intent_api_key` (Streamlit session state, set on Settings page) —
  drives intent classification and the refinement generator. Intent and
  refinement are conceptually classification-shaped; the same key fits.
- `OPENAI_API_KEY_CONTENT_GENERATION` — Stage 6 LLM call (the actual write).
- `OPENAI_API_KEY_EVALUATION` — Stage 7 LLM judges (Tier 2). Tier 1 always
  runs regardless of this key.
- `OPENAI_API_KEY_TRANSLATION` — only the translation pipeline. Untouched.
