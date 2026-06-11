# Content KPI Inventory — workbook analysis

Detailed findings from `data/Content KPI inventory_AISO.xlsx`, ABN AMRO's
authoritative content-quality KPI catalogue for the AISO (AI Search
Optimisation / AI-assisted content) programme. This document captures *what
is in the file* — taxonomy, schema, vocabularies, blocking gates — so the
implementation choices in [`evaluation_stage.md`](evaluation_stage.md) can be
traced back to the source.

Last updated: 2026-05-15.

---

## 1. Workbook layout

Six sheets. Three carry the substance.

| Sheet | Cell range | Purpose |
|---|---|---|
| `Overview` | empty | Placeholder. |
| `Criteria Quality SEO Accessib.` | `A1:AA51` | Five side-by-side criteria blocks (Quality / SEO COECD / SEO EC&E / **SEO check of generated content** / Accessibility). Most rows reference Textmetrics rules. |
| `Blad3` | `A8:C17` | Small PowerBI pivot — confirms which Blocking KPIs flow to the dashboard. |
| `Inventory` | `A1:ER308` (used range `A1:CE169`) | **Master KPI catalogue.** 132 leaf KPIs + 22 clusters + 7 categories. |
| `Input inventory` | `A1:J92` | Controlled vocabularies (dropdown sources) used by `Inventory`. |
| `Channels and stakeholders` | `A1:I31` | 28 channels × owner / dashboard / guideline URL. |

---

## 2. The Criteria sheet — five horizontal blocks

The sheet stacks five evaluation blocks horizontally. Each block is ~7
columns wide and uses a parallel column scheme so every rule can be mapped
back to the prompt/guardrail library.

### 2.1 Quality criteria — columns A–H

30 rows, sourced from generic Textmetrics rules + ABN AMRO's *Schrijfwijzer*.
Columns:

| Col | Field | Notes |
|---|---|---|
| A | `Rule` | Textmetrics rule slug (e.g. `SentenceLengthLanguageLevel`, `PassiveVoiceLanguageLevel`, `ReadingLevel`, `Jeukwoorden`, `BlackList`, `MandatoryList`). |
| B | `PDFS` | All v — also flagged in the PDFS criteria document. |
| C | `Base (input Textmetrics)` | Free-text description of what is checked. |
| D | `Rule (Textmetrics)` | Specific threshold (e.g. *B1: max 15 words per sentence*). |
| E | `Mapping guardrail/prompt library` | One of the macro categories (Taalgebruik en stijl, Inclusiviteit, …). |
| F | `Mapping tag` | Sub-category (e.g. Zinslengte, Schrijfstijl, Specifieke spelling). |
| G | `Subgroep` | Finer-grained tag where applicable. |
| H | `Meegenomen in quality KPI` | v / x / "v niet geimplementeerd" / "x wordt niet gebruikt". |

Sample rules:

- `SentenceLengthLanguageLevel` — B1: max 15 words per sentence.
- `PassiveVoiceLanguageLevel` — B1: max 0 passive sentences.
- `ReadingLevel` — default B1.
- `AdjectivesLevel` — < 12% of words.
- `DenseText` — paragraph length ≤ 100 words.
- `Format - Money` — `€ 1.000,-` style.
- `BulletPointsLevel` — at least 1 bullet list, 3–6 items.
- `GenderLevel` — neutral default.
- `Schrijf inclusief en divers` — hij/zij, meneer/mevrouw checks.
- `Jeukwoorden` — bank-specific jargon list.
- `BlackList` / `MandatoryList` — word lists (not shipped with the workbook).
- `Leeftijdsdiscriminatie` — direct + indirect age discrimination.

### 2.2 SEO criteria COECD — columns I–O

~30 Textmetrics SEO rules, grouped by surface they apply to: **URL**,
**Paginatitel (H1)**, **Metadescription**, **Tussenkoppen** (H2-H6),
**Linktekst**, **Afbeeldingen — alt tags**, **Unieke content**.

Examples:

- `BodyKeywordDensitySuggestion` — keyword density depends on length.
- `BodyKeywordInFirstWords` — main keyword in first 50 words.
- `TitleNrCharsSuggestion`, `TitleNrKeywordsSuggestion`, `TitleNrWordsSuggestion`.
- `MissingImgAltCheck` — every image must have alt text.
- `NoBrokenLinksExternal` / `NoBrokenLinksInternal` — 4xx/5xx detection.
- `UniqueContent` / `UniqueDescription` / `UniqueTitle` — no duplicates.
- `UrlCheck` / `UrlLength` / `UrlNrKeywords` — URL composition.

### 2.3 SEO criteria EC&E — columns P–R

Manual checklist questions (Confluence-sourced). Examples:

- *"Does the URL describe the page and does the most important keyword appear in it? Shorter is better."*
- *"Is the main keyword in the title (front), H1 and meta description?"*
- *"Do the subheads (H2 / H3) contain relevant keywords and/or synonyms?"*
- *"Do the images have ALT tags?"*
- *"Are you sure no page on the same topic already exists?"*
- *"Has the page got links to other relevant pages on AA.nl?"*
- *"Has the page got links from other pages on AA.nl?"*

### 2.4 SEO check of generated content — columns T–U

**The block specifically about AI-generated content.** Five rubrics from the
*SEO - Content & AI - Search Quality Raters.pptx* deck. These are exactly the
five LLM-judge dimensions any GenAI content layer needs:

1. **Unique, added value in content for user** — `[Missing category]`
2. **Demonstrable expertise / customer experiences in content** — `[Missing category]`
3. **No paraphrased content** — `[Missing category]`
4. **No filler content / too generic information** — `[Missing category]`
5. **Check of generated content by expert** — `[Editorial process]`

The `[Missing category]` tag means these rubrics are not yet mapped to the
prompt-library vocabulary — they're new requirements for the GenAI flow.

### 2.5 Accessibility criteria — columns W–AA

Header only; the full accessibility content lives in the `Inventory` sheet
under the **Accessibility & inclusion** cluster (see §3).

---

## 3. The Inventory sheet — master catalogue

300 rows in the used range; rows 9–169 carry data. After parsing we end up
with:

- **7 top-level categories** (`Quality KPI type = KPI category`)
- **22 clusters** (`Quality KPI type = KPI cluster`)
- **132 leaf KPIs** (`Quality KPI type = KPI`)

Plus 2 synthetic KPIs we add at build time for the GenAI rater rubrics that
are documented in the Criteria sheet but missing from `Inventory` (see §6).

### 3.1 Header rows

| Row | Content |
|---|---|
| 1 | `Content quality KPIs for human-authored & generated content` |
| 3-4 | Jira links to user stories per channel. |
| 6-7 | Block headers — channel groupings. |
| 8 | **Column headers** (the data schema, see §3.3). |
| 9-169 | One KPI / cluster / category per row. |

### 3.2 The 7 top-level categories

| # | Category | Default weight | Contribution |
|---|---|---|---|
| 1 | **Accessibility & inclusion** | High | Content that meets guidelines for inclusion, readability & accessible structuring & design. |
| 2 | **Compliancy & substantive quality** | High | Content that meets guidelines for compliancy & substantive quality. |
| 3 | **(Source) content management** | Medium | (GenAI source) content that meets content management guidelines for metadata and modelling. |
| 4 | **Engagement** | Medium | Content that is improved based on insights related to customer engagement. |
| 5 | **Generic quality check** | High | Ensuring that content is in line with all relevant quality standards. |
| 6 | **Language** | Low | Content that meets guidelines for language. |
| 7 | **Online findability and visibility** | Medium | Content that meets guidelines for online findability & visibility in (AI powered) search results (AEO/GEO/SEO). |

### 3.3 The 22 clusters

| Category | Cluster | Weight |
|---|---|---|
| Accessibility & inclusion | Inclusiveness | Medium |
| | Readability | High |
| | Accessible structuring & design | High |
| Compliancy & substantive quality | Accuracy, efficacy and reliability | High |
| | Lawfullness | **Blocking** |
| Content management | Administrative metadata | Medium |
| | Descriptive metadata | Medium |
| | Modelling | High |
| Engagement | Conversion | Medium |
| | Customer satisfaction & feedback | High |
| | Traffic | Medium |
| Generic content quality check | Content quality assessment | High |
| | Guardrails | High |
| Language | Spelling | Low |
| | Brand voice | Medium |
| | Word choice & writing style | Medium |
| Online findability & visibility | Answer engine optimization | Medium |
| | Content length | Medium |
| | Content structure | Medium |
| | Content value | Medium |
| | Keywords | Medium |
| | Links | Medium |

### 3.4 The KPI-row schema (column 1–31)

Every leaf KPI carries the following fields. This is the schema the
implementation mirrors verbatim in `backend/evaluation/data/kpi_catalogue.json`.

| Col | Field | Sample values |
|---|---|---|
| 1 | `Number` | 01–168 |
| 2 | `Quality KPI type` | `KPI` / `KPI cluster` / `KPI category` |
| 3 | `Primary category/cluster` | e.g. `Compliancy  & substantive quality / accuracy, efficacy & reliability` |
| 4 | `Secondary category/cluster` | optional cross-cutting cluster |
| 5 | `KPI Category name 1` | normalised category name |
| 6 | `KPI Cluster Name 1` | normalised cluster name |
| 7 | **`Final name quality KPI`** | the canonical name (used to derive `kpi_id`) |
| 8 | `Monitoring` | `Mandatory` / `Optional` |
| 9 | **`Weight`** | `Blocking` / `High` / `Medium` / `Low` (one typo `Mediun` normalised) |
| 10 | `Contribution to quality` | one-sentence definition |
| 11 | `Norm for KPI (generic/web)` | threshold or rule (e.g. `B1: max. 15 words per sentence`, `obligatory for all knowledge`, `min. 1-3 referrals`) |
| 12 | `Source for norm (generic/web)` | citation |
| 13 | `Deviant norm (chat)` | channel-specific override |
| 14 | `Source for deviant norm (chat)` | |
| 15 | `Deviant norm (messages)` | channel-specific override |
| 16 | `Source for deviant norm (messages)` | |
| 17 | `Measurement` | how it is measured (e.g. *"degree of foundation of generated content on GenAI knowledge source"*) |
| 18 | **`Indicator`** | the output scale phrase (e.g. *"numerous errors, several errors, moderate errors, few errors, no errors"*) |
| 19 | `Match with existing automated measurement` | `full match` / `limited match` / `no match` / `no match (manual check only)` / `future match (dCLP)` |
| 20 | `Relevance for human-authored /scripted content` | applicability flag |
| 21 | `Relevance for GenAI assisted knowledge management` | applicability flag |
| 22 | `Relevance for instantly generated content` | applicability flag |
| 23–27 | `Relevance for web / chat / messages / employee / app_IB` | channel applicability |
| 28 | `Mapping on guardrail/prompt library category` | hook into the prompt-library macro category |
| 29 | `Comments on mapping of categories` | |
| 30 | `Mapping on guardrail/prompt library tag` | hook into the prompt-library fine tag |
| 31 | `Comments on mapping of tags` | |
| 32–83 | per-channel `Human-authored / Generated / Dashboard / Comment` blocks | wired to existing PowerBI sources. |

### 3.5 Quantitative observations

- 132 leaf KPIs in total.
- **All 132 KPIs are `Applicable` for GenAI-assisted knowledge management.**
- 103 of the 132 are also applicable for "instantly generated" content; the
  remaining 29 only apply when the content has a *GenAI source* (i.e. went
  through a RAG/dCLP step).
- Monitoring × weight distribution (leaf KPIs only):

| Monitoring | Weight | Count |
|---|---|---|
| Mandatory | **Blocking** | 10 |
| Mandatory | High | 27 |
| Optional | High | 30 |
| Optional | Medium | 49 |
| Optional | Low | 16 |

---

## 4. The 10 Blocking KPIs (the hard gate)

A generated piece must pass these for publication. From the inventory:

| # | KPI | Indicator | Measurement | Auto-match status |
|---|---|---|---|---|
| 1 | **Factuality & truthfullness** | `numerous → few → no errors` | count of substantial errors in (generated) content | no match (manual check only) |
| 2 | **Truthfullness** | `many / few / no deviations` | count of deviations from truthfullness standards (e.g. usage of *advisor* in execution-only flows) | no match |
| 3 | **Relevancy** | `off-topic → highly relevant` | degree of relevancy of content to user | no match (manual check only) |
| 4 | **Privacy and security** | `many / few / no deviations` | count of deviations against privacy & security policies | no match |
| 5 | **Lawfullness** (cluster) | `low / medium / high maturity` | content audit on law & administration KPIs | no match |
| 6 | **Human expert check (substance)** | `completed step yes/no` | info-owner approval (dCLP step) | future match (dCLP) |
| 7 | **Human expert check (compliancy & legal)** | `completed step yes/no` | PO + Legal & Compliance approval (dCLP step) | future match (dCLP) |
| 8 | **Human expert check (content)** | `completed step yes/no` | second content-specialist approval (dCLP step) | future match (dCLP) |
| 9 | **Status of evaluation** | `completed step yes/no` | re-checked within last 12 months (dCLP step) | limited match (no norm/no dCLP) |
| 10 | **Tracability** | `used / not used` | source ID + version tag present in content | no match |

Asymmetry that drives the implementation design:

- **5 are deterministic-checkable** (4 dCLP step flags + tracability).
- **5 are inherently judgemental** (factuality, truthfullness, relevancy, privacy deviations, lawfullness audit) → need LLM-as-judge.

Plus an 11th Mandatory-High KPI worth flagging: **Approved source content for GenAI** (`exclusion / no exclusion` indicator) — every GenAI source must carry an "exclusion tag" if it must not be used.

---

## 5. The indicator scales (the output vocabularies)

Every leaf KPI carries one indicator phrase in column 18. There are 63
distinct phrases. The most common — and the ones the evaluation layer needs
strong typing for:

| Indicator phrase | # KPIs | Enum we map it to |
|---|---|---|
| `present, not present` | 41 | `PresenceScale` |
| `low, medium or high maturity` | 29 | `Maturity` |
| `yes/no deviation from norm` | 21 | `DeviationYesNo` |
| `completed step yes/no` | 4 | `YesNoScale` |
| `very low, low, medium, high, very high` | 2 | `FivePointScale` |
| `right length, too long` | 2 | `LengthScale` |
| `unclear, somewhat clear, clear, very clear` | 2 | `ClarityScale` |
| `very incomplete → fully complete` (5-point) | 2 | `CompletenessScale` |
| `many deviations, few deviations, no deviations` | 1 | `DeviationScale` |
| `many ambiguities, few ambiguities, no ambiguities` | 1 | `AmbiguityScale` |
| `off-topic → highly relevant` (4-point) | 1 | `RelevanceScale` |
| `numerous errors → no errors` (5-point) | 1 | `ErrorScale` |
| `no grounding → fully grounded` (4-point) | 1 | `GroundednessScale` |
| `applicable, not applicable` | 2 | `ApplicableScale` |
| `used, not used` | 2 | `UsedScale` |
| `A1, A2, B1, B2, C1, C2` | 1 | `LanguageLevelScale` |
| `exclusion, no exclusion` | 1 | `ExclusionScale` |
| 5-point CES / CSAT / 11-point NPS | 1 each | `CESScale` / `CSATScale` / `NPSScale` |

38 leaf KPIs use bespoke value-enumeration phrases (e.g.
`e.g. consumer clients, affluent clients, business clients`,
`e.g. agriculture, energy, food`) — these are metadata enumerations rather
than evaluation indicators, so they're not mapped. The catalogue records the
raw phrase under `indicator_phrase` for traceability and leaves `indicator`
as `null` for those rows.

---

## 6. Guardrail / prompt-library mapping

Columns 28–31 connect each KPI to ABN AMRO's prompt-library vocabulary.
Distribution today (across the 132 leaf KPIs):

| Guardrail category | # KPIs |
|---|---|
| `[Missing category]` | 34 |
| `[Metadata]` | 24 |
| `Inhoud & Duidelijkheid` | 17 |
| `Taalgebruik en Stijl` | 16 |
| `[meerdere, zie comments]` | 11 |
| `Leesbaarheid en Toegankelijkheid` | 10 |
| `Structuur en Opmaak` | 10 |
| `[Multiple categories]` | 8 |
| `Communicatie en Interactie` | 7 |
| `Branding en Merkidentiteit` | 5 |
| `Inclusiviteit en Genderneutraliteit` | 4 |
| `[Editorial process]` | 4 |
| `[Not applicable]` | 4 |
| `Digitale Interacties` | 2 |
| `Terminologie en Consistentie` | 1 |

The 19 distinct *tags* under those categories — `Specifieke spelling`,
`Schrijfstijl`, `Zinslengte`, `Inclusiviteit`, `Helderheid`, `Titels`,
`Expertise`, `Klantgericht`, `Vertalingen`, … — are the same tag space used
by AURORA's PageIndex retrieval enrichment (see `docs/pageindex_scale_up_plan.md`).
That is the closed loop: the evaluation result on a KPI tagged
`Inhoud & Duidelijkheid` can route back to the prompt-refinement stage's
prompts in the same category.

---

## 7. The `Input inventory` sheet — controlled vocabularies

10 columns of dropdown sources. Independent — each column is its own list,
the row indices are nominal (column A just enumerates 1–88 for spreadsheet
sorting).

| Col | Vocabulary | n |
|---|---|---|
| 2 | KPI clusters (cross-cutting names like *Content quality – brandvoice*) | 21 |
| 4 | Type | 7 (`KPI category`, `KPI cluster`, `KPI`, `OKR`, `Value`, `KPI/Filter`, header) |
| 6 | Human-authored / Generated applicability | 11 (`Applicable`, `Need`, `Not applicable`, `Irrelevant`, `[See Conversational]`, `Only applicable for GenAI source`, …) |
| 8 | Guardrails and prompts categorisation | 19 |
| 10 | Guardrails and prompt tags | 91 fine-grained tags |

The applicability vocabulary in column 6 is what drives the evaluation
layer's `applicable(origin, channel)` filter — a KPI applies when its
relevance value matches *Applicable\** or *Need*; everything else (Irrelevant
/ Not applicable / Only applicable for GenAI source when origin is `instant`)
filters it out.

---

## 8. The `Channels and stakeholders` sheet

28 rows, one per channel × owner pairing. Captures:

- **Channel** — `Web - pathfinder`, `Web - articlepages`, `Web - foutmeldingen`, `Web - storingsmeldingen`, `Web - instructies`, `Web - productpages`, `Web - blogs`, `Web - Conversion`, `Web - Generated`, `Web - SEO`, `Conversational`, `Messages - campaign`, `Messages - system`, `Messages - advisor`, `Employee`, `Digital assets - images / docs / forms`, `App IB/applicative`, `Social media`, `Voice`, plus generic rows for `Generic`, `Generic - Content`, `Generic - SEO/SEA`, `Generic - Language`, `Generic - GenAI`, `Generic - Accessibility (OMO)`.
- **Stakeholder** — single owner per row.
- **Dashboard, tooling and sources** — e.g. `Content dashboard (Tridion, DAM, website), content alert dashboard (Tridion) in PowerBI`, `CIED-Chatbots dashboard in PowerBI (MSCP). Azure Monitor (AI generated content)`.
- **Guidelines** — Confluence URLs.
- **Organisation** — `COE CD, cirkel 1` / `cirkel 2` / `cirkel 3` / `EC&E` / `Daily Banking & NCTO (CRO team)` / `Wealth` / `BMC` / `Textmetrics`.
- **Meeting / Status collection / Status review / Comment** — operational metadata.

Two takeaways for AURORA:

1. The evaluation results need to land in the **PowerBI Content Dashboard**
   (the Web-channel rows all converge on it). Using the standardised
   indicator-enum values rather than free-text scores is what makes that
   one-shim integration possible.
2. The **`Web - Generated`** row (Mark Westbeek / Xaviera Ringeling) is the
   most relevant stakeholder pairing for AURORA's editorial-co-pilot path.

---

## 9. The `Blad3` sheet — what is genuinely blocking

A small PowerBI pivot showing which Application × KPI combinations carry the
**Blocking** flag in the live dashboard today. Confirms only two clusters
flag rows as `Blocking` in the dashboard:

- **Compliancy & substantive quality** (2 rows: *Accuracy, efficacy and reliability* + *Human expert check (substance)*)
- **Lawfullness** (2 rows: *Lawfullness* + *Human expert check (compliancy & legal)*)

The other 6 Blocking rows that show up in the `Inventory` sheet
(Factuality, Truthfullness, Relevancy, Privacy, dCLP signoff steps, Tracability)
are weighted as Blocking but not yet wired into the dashboard's Blocking
filter. That's a gap to close as the evaluation layer ships real signal.

---

## 10. Implications that shaped the implementation

In one paragraph each:

**Three-tier evaluator.** The 5 deterministic-checkable Blocking KPIs need
a code path that runs on every generation without LLM cost; the 5
judgement-based Blocking KPIs need an LLM-as-judge tier; the 4 dCLP-step
KPIs need a workflow flag tier. So the evaluator is three tiers, not one.

**Indicator enums, not floats.** Every KPI already has a finite output
scale that the bank's dashboards read. Emitting enum values (rather than
numeric "scores" we'd then have to map back) lets the evaluator's output
plug into the existing dashboards as-is.

**Origin + channel filter.** All 132 KPIs apply to *some* combination of
origin (`human` / `genai_knowledge` / `instant`) × channel (`web` / `chat` /
`messages` / `employee` / `app_ib`). The applicability matrix is in the
workbook; we materialise it via `Catalogue.applicable(origin, channel)` so
running the evaluator with the wrong channel doesn't fire irrelevant KPIs.

**Closed loop to prompt refinement.** Every leaf KPI is tagged with a
prompt-library category + tag. When an evaluation result flags a KPI
failure, the prompt-refinement stage can target prompts in the same tag
space — the workbook is the contract between the two stages.

**Synthetic KPIs for the GenAI rater rubrics.** The five
"Search Quality Raters for GenAI" rubrics in the Criteria sheet (T/U) are
not in the Inventory sheet. Two of them (`No paraphrased content`,
`No filler content / too generic information`) had no Inventory row at all,
so the build script synthesises them as Optional-High KPIs under
*Online findability and visibility / content value* so the LLM-judge tier
has catalogue rows to bind to. This is the only place the implementation
*extends* the workbook rather than mirroring it.
