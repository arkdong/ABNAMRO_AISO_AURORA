# AURORA Evaluation — What Every Check Does and Where It Comes From

This document explains the three-tier KPI evaluator that ships in
`aurora_tool_server/aurora_tool_server/evaluation/`. Every check is traced to
its source in ABN AMRO's own governance workbook, **`Content KPI
inventory_AISO.xlsx`** (the COECD/AISO content-quality inventory), so that a
reviewer can verify each gate against the bank's stated norm rather than
trusting the code.

---

## 1. The KPI catalogue

The evaluator does not invent quality rules. It loads
`assets/evaluation/kpi_catalogue.json`, a build-time export of the workbook's
**Inventory** sheet produced by `demo/rag/scripts/build_kpi_catalogue.py`:

- **135 KPI entries** = 132 leaf KPIs from Inventory rows 9–169 + 3 synthetic
  entries from the workbook's *Criteria* sheet ("Search quality raters for
  GenAI" block), marked `fields_inferred: true` because the workbook gives
  them a name but no weight/norm/relevance of their own.
- **7 categories** and **22 clusters**, exactly as the workbook defines them.
  Category names are canonicalized to the category-row spelling (the workbook
  spells the same category differently in different columns, which would
  otherwise split the maturity rollups).
- Each entry carries the workbook's own vocabulary: `weight`
  (Blocking/High/Medium/Low), `monitoring` (Mandatory/Optional), `norm` (plus
  channel-deviant norms for chat and messages), `measurement`, `indicator`
  (the standardised result scale), per-origin and per-channel `relevance`
  cells, and guardrail tags.
- **Provenance fields** added from previously-unused workbook columns: the
  hierarchical KPI `number` (e.g. `01.02.03`), `norm_source` (often a Ruby /
  Confluence or SEO-checklist URL), and `tool_rule`/`tool_dashboard` — the
  Textmetrics rule name and PowerBI dashboard the KPI is bound to in the
  bank's existing tooling.

**The Blocking set.** Exactly 10 KPIs are `Blocking` + `Mandatory` in the
workbook, and the catalogue matches it 1:1: factuality & truthfullness,
truthfullness, relevancy, privacy & security, tracability, approved source
for GenAI, status of evaluation, and the three human expert checks
(substance; compliancy & legal; content). A failed Blocking KPI sets
`passed=False` and is named in `failed_blocking`.

**Origin and channel relevance.** Every evaluation declares an `origin`
(`human` | `genai_knowledge` | `instant`) and a `channel` (`web` | `chat` |
`messages` | `employee` | `app_ib`). The workbook's relevance columns decide
which KPIs apply; the evaluator honours them in all tiers. Cells reading
*"Only applicable for GenAI source"* (tracability, approved-source) are
treated as **conditional**: the checks still run on `instant` content and
no-op gracefully when there are no citations — skipping two Blocking source
checks on the default origin would silently disable the very gate they exist
for.

---

## 2. The three tiers

| Tier | What | Cost | When it runs |
|---|---|---|---|
| **1** | 18 deterministic Python checks | free, repeatable | always |
| **2** | 13 schema-constrained LLM judges | ~13 small model calls | only if an evaluation API key + model are configured |
| **3** | 4 dCLP human-signoff requirements | none (declared, not executed) | per relevance |

**Flow** (`service.py: evaluate_draft`):

1. Run Tier 1. If any **Mandatory + Blocking** Tier-1 KPI fails, return
   immediately with `passed=False` and the offending IDs — no LLM budget is
   spent on content that already failed a hard gate.
2. Run Tier 2 (the judges that apply to this origin/channel) in parallel
   (thread pool, max 6). Without an API key each judge emits a
   `not_evaluated` record that passes permissively; with `strict_mode=true`
   those skipped records are flipped to failing so missing infrastructure
   fails closed instead of silently approving.
3. Declare Tier-3 dCLP steps that apply. They are reported in
   `dclp_steps_required` and as pending tier-3 results, **but they do not
   gate `passed`** — they are workflow signoff flags, not content verdicts
   (e.g. *status of evaluation* is a 12-month lifecycle re-check; blocking
   every fresh draft on it would be a category error). AURORA flags human
   steps; it never auto-clears them, and it never claims they are done.
4. Aggregate `maturity_by_category` (pass-ratio per workbook category →
   `low` < 50% ≤ `medium` < 80% ≤ `high`) — the rollup shape the bank's
   PowerBI dashboards already use.

Every `KPIResult` carries: the catalogue slug and name, cluster/category,
weight, monitoring, the **indicator scale name and enum value** (the
workbook's own vocabulary, so results merge into existing dashboards
unchanged), a `raw_metric` payload with the underlying numbers, a one-line
reason, the tier, and whether the value came from `deterministic`, `llm`, or
was `skipped`.

---

## 3. Tier 1 — the 18 deterministic checks

Each row: the catalogue slug, the workbook KPI number and norm (quoted), and
what the code actually measures. Norm-source abbreviations: *Ruby* = the
bank's Schrijfwijzer/Confluence pages; *TM* = Textmetrics rule (the bank's
existing content-checking tool); *SEO* = AAB SEO-team checklist.

### Readability (Accessibility & inclusion)

| Check | Workbook | Norm (quoted) | What the code does |
|---|---|---|---|
| `sentence_number_of_words` | 01.02.05 · High/Optional · Ruby+TM `SentenceLengthLanguageLevel` | "min. 5 words, max. 20 words/B1: max. 15 words"; chat: "Max 10-12 words per sentence" | Splits sentences, counts words per sentence; fails when >10% of sentences are over 15 words (12 on chat) or under 5 words. *The 10% tolerance is engineering calibration, documented as such — the workbook states the per-sentence bounds only.* |
| `paragraph_bubble_number_of_words_sentences` | 01.02.02 · High/Optional · TM `Densetext` | "max. 100 words per paragraph"; chat: "max. 160 characters per bubble / max 3 bubbles" | Web: every paragraph ≤100 words. Chat: every bubble ≤160 characters and at most 3 bubbles. |
| `text_number_of_sentences` | 01.02.08 · High/Optional · TM | "B1: max. 100 sentences" | Counts sentences in the body; fails above 100. |
| `reading_level` | 01.02.03 · **High/Mandatory** · Ruby+TM `Reading level` | "B1" | Computes Flesch reading ease and maps it to a CEFR band; A1/A2/B1 pass. *The Flesch→CEFR cut-offs (90/80/60/50/30) are calibration, not a workbook norm, and the formula is English-tuned — which is why the same KPI also has a Tier-2 CEFR judge as the authoritative signal.* |

### Writing style (Language)

| Check | Workbook | Norm | What the code does |
|---|---|---|---|
| `writing_style_active` | 06.03.04 · **High/Mandatory** · Ruby+TM `PassiveVoiceLanguageLevel` | "B1: max. 0 passive sentences" | Regex passive-voice detection (EN auxiliary+participle; NL *worden/wordt/werd…* per intent language). Any hit fails — the workbook norm is zero. Hit count stays in `raw_metric`. |

### Structure & design (Accessibility & inclusion)

| Check | Workbook | Norm | What the code does |
|---|---|---|---|
| `bullet_list_points` | 01.03.02 · High/Optional · "Guardrail opsommingen" | "min. 3 - max. 6 points in bullet list" | Finds contiguous bullet runs; requires ≥1 list and every list to have 3–6 items. (The Textmetrics rule only checks "a list exists"; the workbook itself flags that as "limited match — no norm for number of items", so the real norm is enforced here.) |
| `images_with_missing_alt_text` | 01.03.07 · **High/Mandatory** · Ruby+TM `Images with missing alt text` | "obligatory alt tag in case of informative image" | Every markdown image must carry non-empty alt text; no images = vacuous pass. The KPI measures the *defect*, so the emitted value follows the workbook orientation: `not_present` (no missing-alt images) = pass. Markdown can't distinguish informative from decorative images, so all are held to the norm. |
| `headers_and_titles` | 01.03.06 · High/Optional · Ruby | "max. 45 characters incl. space / 3-8 words" | Every heading (H1–H6) must be ≤45 characters and 3–8 words. |

### SEO & findability (Online findability and visibility)

| Check | Workbook | Norm | What the code does |
|---|---|---|---|
| `h1_header_presence` | 07.03.01 · Medium/Optional · SEO+TM | "1 H1 header" | Exactly one `#` heading. |
| `h1_header_keywords` | 07.05.03 · **High/Mandatory** · TM `H1NrKeywordsSuggestion` | "1-2 keywords" | Counts intent topic-keywords in the H1; passes at 1–2 (keyword stuffing fails the upper bound). Emits the workbook's `DeviationYesNo` scale. Skipped (not failed) when the caller supplied no intent keywords. |
| `h2_6_headers_number` | 07.03.02 · Medium/Optional · TM | "1-14 headings" | Counts H2–H6 headings; passes 1–14. |
| `text_number_of_words` | 07.02.05 · Medium/Optional · TM | "min. 300 words" (web) | Word count ≥300. Chat is excluded by the workbook's relevance cell, not by code. |
| `body_content_key_word_density` | 07.05.01 · Medium/Optional · TM | "min. 1 keyword for 100-199 words; 2 for 200-1499; 3 for ≥1500" | Counts keyword occurrences against the graduated minimum for the body length. Skipped without intent keywords. |
| `body_content_key_words_in_first_words` | 07.05.02 · Medium/Optional · TM `BodyKeywordInFirstWords` | "min. 1 main keyword" (first 50 words per the TM rule) | At least one topic keyword in the opening 50 words. Skipped without intent keywords. |
| `referrals` | 07.04.04 · **High/Mandatory** · SEO `HasReferrals` | "min. 1-3 referrals" | Counts markdown links in the body (referrals to related content); passes 1–3. Citation markers `[n]` are evidence references and do not count. |

### Compliance source gates (Blocking floors)

| Check | Workbook | Norm | What the code does |
|---|---|---|---|
| `tracability` | 03.01.08 · **Blocking/Mandatory** | "obligatory for all GenAI source content" — measurement: "source ID and version tag in content" | The draft must carry structured citations (`source_doc::node_id` pairs back to corpus snippets) — the operational stand-in for "source ID + version tag" until a versioned source manifest exists. No citations = Blocking fail. |
| `approved_source_content_for_genai` | 03.02.01 · **Blocking/Mandatory** | "all GenAI source content which is not allowed for GenAI application has an exclusion tag" | Looks up every cited snippet; if any carries `exclude_for_genai: true` (now a real field on the `Snippet` schema), the draft is Blocking-failed with the offending source IDs named. |
| `factuality_truthfullness` (floor) | 02.01.04 · **Blocking/Mandatory** · workbook: "no match (manual check only)" | "no substantial errors in knowledge" | Deterministic *floor* under the Tier-2 factuality judge: every `[n]` citation marker in the body must map to a real retrieved snippet. A fabricated citation index is the cheapest provable factuality violation and Blocking-fails without any LLM call. This range check is an engineering floor, declared as such — the full judgement is Tier 2's. |

---

## 4. Tier 2 — the 13 LLM judges

Each judge scores exactly one rubric and is **schema-constrained**: the
OpenAI structured-output call is bound to a Pydantic model whose `value`
field is the KPI's indicator enum — the model literally cannot return a
value outside the workbook's scale. Judges receive the user query, the
retrieved evidence snippets, the draft, the rubric, **and the workbook norm
for the KPI (channel-specific where the workbook defines one)**. Judges run
in parallel; one judge erroring marks only its own KPI as failed
(`judge error`), never the batch. `passed` derives from the per-scale
passing sets in `indicators.py`.

| Judge | KPI (workbook #) | Weight | Scale (workbook indicator) | Passes on | What it asks |
|---|---|---|---|---|---|
| `factuality` | factuality_truthfullness (02.01.04) | **Blocking** | ErrorScale ("numerous…no errors") | `none` only — the norm is "no substantial errors"; *few errors* does not pass a Blocking gate | Count claims contradicting or unsupported by the retrieved evidence. |
| `truthfullness` | truthfullness (02.01.09) | **Blocking** | DeviationScale | `none` | Deviations from bank standards for truthful substance (e.g. "advice" for execution-only, invented guarantees, dramatised risk/return — the workbook's own example). |
| `relevancy` | relevancy (02.01.08) | **Blocking** | RelevanceScale | `reasonable`/`highly` (norm: "at least reasonable relevant") | Does the draft address the user's actual query? |
| `privacy_security` | privacy_and_security (02.02.02) | **Blocking** | DeviationScale | `none` | Policy deviations: personal data exposure, non-AAB external links, BSN/IBAN mishandling. Channel-deviant norms (e.g. "no attachments in chat") reach the judge via the norm block. |
| `groundedness` | groundedness_source (02.01.05) | High/Mand. | GroundednessScale | `reasonable`/`full` (norm: "at least reasonable grounding") | Share of substantive claims traceable to the snippets. Workbook marks this *Not applicable* for human-authored content — the judge does not run for `origin=human`. |
| `completeness_source` | completeness_source (02.01.02) | High/Opt. | CompletenessScale | `mostly`/`full` | Coverage of the snippet substance relevant to the query. Also origin-filtered (not for `human`). |
| `comprehensiveness` | comprehensiveness_answer (07.01.02) | High/Mand. | CompletenessScale | `mostly`/`full` (norm: "min. mostly complete") | AEO-style: does the draft answer the *entire* query including natural follow-ups? |
| `clarity` | clarity (02.01.01) | High/Opt. | **AmbiguityScale** ("many/few/no ambiguities" — the workbook's indicator for this KPI) | `none` (norm: "no ambiguities") | Count places a reader could reasonably take a different meaning. |
| `reading_level` | reading_level (01.02.03) | High/Mand. | LanguageLevelScale (A1–C2) | A1/A2/B1 (norm: "B1") | CEFR judgement of the draft — the authoritative signal over the Tier-1 Flesch floor, whose cut-offs are calibration. |
| `uniqueness_added_value` | unique_added_value (Criteria sheet T8, synthetic) | High/Opt.* | PresenceScale | `present` | Does the draft add value beyond restating the obvious/snippets? *Bound to its own Criteria-sheet entry — **not** to workbook KPI 07.04.01 "Body content - uniqueness", which is a corpus-deduplication measurement with opposite polarity and stays un-evaluated until a content-collection index exists.* |
| `demonstrable_expertise` | experience_expertise (07.04.02) | Medium/Opt. | PresenceScale | `present` | Concrete experience/examples/customer-context vs generic marketing copy (Criteria sheet T9 ≈ workbook 07.04.02). |
| `no_paraphrase` | no_paraphrase (Criteria sheet T10, synthetic) | High/Opt.* | PresenceScale | `present` | Does the draft go beyond reshuffling snippet sentences? |
| `no_filler` | no_filler (Criteria sheet T11, synthetic) | High/Opt.* | PresenceScale | `present` | Free of filler ("It's important to consider…", "In today's world…")? |

\* Synthetic entries: the Criteria sheet provides the criterion name only;
weight/monitoring/norm/relevance are builder defaults, flagged
`fields_inferred: true` in the catalogue.

---

## 5. Tier 3 — the 4 dCLP human-signoff steps

Four workbook KPIs use the indicator *"completed step yes/no"* — they record
whether a human stage of the bank's **dCLP** (digital content lifecycle
process) has been signed off, and cannot be evaluated by code:

| Step | Workbook | Norm | dCLP stage (norm source) |
|---|---|---|---|
| `human_expert_check_substance` | 02.01.06 · Blocking/Mandatory | "obligatory for all knowledge" | "Stap 4. goedkeuren" (approve) |
| `human_expert_check_compliancy_legal` | 02.02.01 · Blocking/Mandatory | "obligatory for all knowledge" | "Stap 3. controleren" (review) |
| `human_expert_check_content` | 05.01.02 · Blocking/Mandatory | "obligatory for all content" | "Stap 3. controleren" |
| `status_of_evaluation` | 03.01.05 · Blocking/Mandatory | "all content is evaluated each 12 months." | 12-month lifecycle re-check |

The evaluator declares which of these apply to the content's origin/channel
(`dclp_steps_required`) and emits them as pending tier-3 results. They do
**not** gate `passed` — they are workflow state owned by the dCLP system,
not a property of the draft — and AURORA never auto-clears them. For
`origin=instant` the workbook scopes them to the *GenAI source* rather than
the output, so none are required; for `human`/`genai_knowledge` all four are
listed as awaiting signoff.

---

## 6. Changes made against the original implementation (audit log)

The evaluator was audited line-by-line against the workbook before this
port. The following corrections were applied — each one is a place where the
previous code contradicted, under-enforced, or mis-attributed a workbook
norm:

1. **Tier-3 pendings no longer gate** — previously every `human`/
   `genai_knowledge` evaluation structurally failed because pending signoffs
   counted as Blocking failures.
2. **The two Blocking source gates now run on `origin=instant`** —
   previously the *"Only applicable for GenAI source"* relevance cell
   silently excluded tracability and approved-source on the service's
   default origin.
3. **Passive voice re-mapped** from `sentence_structure` (01.02.06, "max. 3
   difficult sentences" — a different phenomenon) to `writing_style_active`
   (06.03.04) with the workbook's zero-tolerance threshold, replacing an
   invented ≤10% ratio.
4. **Factuality pass boundary tightened** — `few errors` no longer passes a
   Blocking gate whose norm is "no substantial errors".
5. **Tier 2 honours the relevance columns** — judges no longer run on
   origin/channel combinations the workbook excludes.
6. **The uniqueness judge detached from `body_content_uniqueness`** — it
   previously recorded an LLM "added value" opinion, with inverted polarity,
   against a Mandatory/High corpus-deduplication KPI.
7. **Clarity judge bound to `AmbiguityScale`** (the workbook's indicator for
   that KPI) with the norm-faithful passing set, instead of a generic
   clarity scale.
8. **`ExclusionScale` passing set fixed** (both values passed before) and
   `Snippet.exclude_for_genai` added, turning the approved-source gate from
   a permanent vacuous pass into a real check.
9. **H1 keyword check enforces both bounds (1–2)** and emits the workbook's
   `DeviationYesNo` indicator; missing keywords skip instead of fail.
10. **Bullet lists checked against the real norm (3–6 items)** instead of the
    presence-only tool rule the workbook itself flags as deficient.
11. **Alt-text indicator orientation corrected** to the workbook's
    defect-present vocabulary so values merge into dashboards unchanged.
12. **Sentence length enforces the min-5-words floor**; remaining tolerances
    are documented as calibration distinct from the norm.
13. **Channel deviant norms are live** — chat evaluations use the workbook's
    bubble norms (≤160 chars, ≤3 bubbles, ≤12-word sentences); judges see
    channel-specific norms in their prompts.
14. **Seven new deterministic checks** added straight from workbook norms:
    sentence count, referrals, H2–H6 count, body word count, heading length,
    keyword density, main-keyword-in-first-words.
15. **A CEFR reading-level judge** added over the Flesch floor.
16. **Catalogue regenerated** with canonical category names (maturity
    rollups no longer split across spelling variants) and provenance fields
    (`number`, `norm_source`, `tool_rule`, `tool_dashboard`,
    guardrail-mapping comments) restored from previously-dropped workbook
    columns.

## 7. Known limits (deliberate, not hidden)

- **Tracability checks source IDs, not version tags** — a versioned source
  manifest is pilot scope.
- **`exclude_for_genai` is enforced but nothing sets it yet** — the corpus
  ingestion pipeline must populate the tag.
- **Un-implemented automatable KPIs** (workbook-sourced, deferred with
  reasons): broken-link checking (needs a link inventory or network access),
  title/meta-description/URL SEO checks (the generation schema has no
  title/meta/URL fields yet), wordlist rules (Jeukwoorden, black/mandatory
  lists — the lists are bank-internal inputs, not in the workbook), corpus
  deduplication for the uniqueness KPIs (needs a content-collection index),
  difficult-sentence detection (low regex precision), engagement KPIs
  (CES/CSAT/NPS — post-publish telemetry, not generation-time).
- **38 of 135 catalogue KPIs have no mapped indicator scale** (free-text
  phrases like "score in %"); their phrases are preserved in
  `indicator_phrase` and they carry no automated gate.
- The deterministic Flesch reading-level floor and the 10% sentence-length
  tolerance are **calibration**, flagged as such here and in the code; the
  workbook norms are the quoted cell texts.
