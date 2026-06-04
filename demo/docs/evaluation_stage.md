# Stage 6 — Content evaluation

Tracks the design and implementation of AURORA's evaluation layer: scoring
generated content against ABN AMRO's Content KPI catalogue. Reads as a
companion to [`kpi_workbook_analysis.md`](kpi_workbook_analysis.md), which
documents the source-of-truth Excel file.

Last updated: 2026-05-15.

---

## 1. What this stage does

Takes the output of Stage 5 (content generation) and returns an
`EvaluationResult` that:

1. **Gates publication** on the 10 Mandatory + Blocking KPIs from the
   workbook. If any fail, the content is `passed=False` with the offending
   `kpi_id`s listed.
2. **Reports per-category maturity** (`low` / `medium` / `high`) over the 7
   top-level categories — the same audit-style rollup the existing PowerBI
   content dashboards consume.
3. **Surfaces required dCLP signoffs** — the 4 editorial-process KPIs that
   can only be cleared by a human checking a box in the workflow system.
4. **Records standardised indicator-enum values** per KPI, so the result
   plugs into the bank's dashboards without re-mapping.

Two modes, mirroring every other AURORA stage:

- **LLM mode** when both `OPENAI_API_KEY_EVALUATION` and an evaluation
  model are configured — Tier 2 LLM-judge rubrics run.
- **Deterministic mode** otherwise — Tier 1 still runs; Tier 2 KPIs are
  recorded as `not_evaluated` and (in default lenient mode) pass.

A `strict_mode=True` flag flips the deterministic-mode default to *fail* —
useful in prod to ensure missing infra cannot silently approve content.

---

## 2. Pipeline placement

```
1. Intent classification
2. Profile selection
3. Initial retrieval (PageIndex)
4. Prompt refinement (iterative)
5. Content generation (LLM with structured output)
6. NEW — Content evaluation (this stage)
```

Auto-runs at the end of Stage 5; the user does not "proceed" into it the
way they proceed between earlier stages — it is the verdict on the just-
generated content.

---

## 3. Module layout

```
backend/evaluation/
├── __init__.py            # public API: evaluate, EvaluationResult, KPIResult, KPI, load_catalogue
├── types.py               # Pydantic shapes (Channel, Origin, KPIResult, EvaluationResult)
├── indicators.py          # 25 indicator enums + PASSING_VALUES default norms
├── catalogue.py           # KPI dataclass + Catalogue accessor (load once, filter by origin/channel)
├── tier1_deterministic.py # 11 deterministic checks registered against real catalogue KPI ids
├── tier2_judges.py        # 12 LLM-judge rubrics, parallel ThreadPoolExecutor
├── tier3_human_loop.py    # dCLP step requirements (declared, not executed)
├── prompt.py              # shared JUDGE_SYSTEM_PROMPT + 12 RUBRICS dict
├── service.py             # evaluate() — orchestrates + short-circuits + aggregates
├── data/
│   ├── kpi_catalogue.json # generated from the xlsx, checked in
│   └── wordlists/         # placeholder for jeukwoorden / blacklist / mandatory wordlists
└── tests/                 # 28 tests (tier1 unit, tier2 mocked-LLM, service orchestration)
```

Plus one build-time script:

- `rag/scripts/build_kpi_catalogue.py` — idempotent xlsx → JSON exporter.

---

## 4. Build-time: KPI catalogue export

```
python rag/scripts/build_kpi_catalogue.py
```

Reads `data/Content KPI inventory_AISO.xlsx`, walks the `Inventory` sheet
rows 9–169, and writes `backend/evaluation/data/kpi_catalogue.json`.

### Schema of `kpi_catalogue.json`

```jsonc
{
  "source_xlsx": "data/Content KPI inventory_AISO.xlsx",
  "generated_at": "2026-05-15T00:54:30+00:00",
  "categories": [{"name": "Compliancy & substantive quality", "weight": "High"}, ...],
  "clusters":   [{"name": "Readability", "weight": "High", "primary_cluster": "Accessibility & inclusion / readability"}, ...],
  "kpis": [
    {
      "id":                "factuality_truthfullness",       // slug of `Final name quality KPI`
      "name":              "Factuality & truthfullness",
      "primary_cluster":   "Compliancy  & substantive quality / accuracy, efficacy & reliability",
      "secondary_cluster": null,
      "category":          "Compliancy  & substantive quality",
      "cluster_short":     "accuracy, efficacy & reliability",
      "monitoring":        "Mandatory",
      "weight":            "Blocking",
      "contribution":      "Accurate, effective and reliable content...",
      "norm":              "no substantial errors in knowledge",
      "norm_chat":         null,
      "norm_messages":     null,
      "measurement":       "count of substantial errors in (generated) content",
      "indicator_phrase":  "numerous errors, several errors, moderate errors, few errors, no errors",
      "indicator":         "ErrorScale",                     // enum class name
      "automated_match":   "no match (manual check only)",
      "relevance": {
        "human":           "Applicable",
        "genai_knowledge": "Applicable",
        "instant":         "Applicable",
        "web":             "Applicable",
        "chat":            "Applicable",
        "messages":        "Applicable",
        "employee":        "Applicable",
        "app_ib":          "Applicable"
      },
      "guardrail_category": "[Missing category]",
      "guardrail_tag":      "[meerdere, zie comments]"
    },
    ...
  ]
}
```

Counts after the last build:

- 134 KPIs (132 from `Inventory` + 2 synthetic GenAI rater entries — see §11).
- 7 categories, 22 clusters.
- 10 Blocking KPIs.
- 38 KPIs with unmapped indicator phrases (mostly metadata enumerations
  that aren't evaluation indicators, e.g. *e.g. consumer clients, affluent
  clients*; these remain `indicator: null`).

The phrase → enum mapping lives in `INDICATOR_ENUM_BY_PHRASE` in the build
script — keep it in lockstep with `backend/evaluation/indicators.py`.

### When to rebuild

Manual today. Run the script whenever:

- The xlsx file is updated by the COECD team.
- A new indicator scale is added that needs an enum.
- The `INDICATOR_ENUM_BY_PHRASE` map is extended.

A CI assertion that the on-disk JSON matches the xlsx hash is the natural
follow-up — not implemented yet.

---

## 5. Type system (`indicators.py` + `types.py`)

### 5.1 Indicator enums

Every workbook indicator phrase that maps to a known scale becomes a
Python `Enum` subclass of `str`. The full registry in
`backend/evaluation/indicators.py`:

| Enum | Values | Maps from |
|---|---|---|
| `Maturity` | low / medium / high | category & cluster audit rollups |
| `PresenceScale` | present / not_present / unknown | the 41-KPI "present, not present" scale |
| `YesNoScale` | yes / no / unknown | dCLP step signoffs |
| `DeviationYesNo` | yes / no / unknown | "yes/no deviation from norm" |
| `DeviationScale` | many / few / none / unknown | truthfullness, privacy |
| `AmbiguityScale` | many / few / none / unknown | Clarity (variant phrasing) |
| `RelevanceScale` | off_topic / somewhat / reasonable / highly | Relevancy KPI |
| `GroundednessScale` | none / limited / reasonable / full | Groundedness (source) KPI |
| `ErrorScale` | numerous / several / moderate / few / none | Factuality |
| `CompletenessScale` | very_incomplete / incomplete / fairly / mostly / full | Completeness, Comprehensiveness |
| `ClarityScale` | unclear / somewhat / clear / very_clear | Clarity KPI |
| `FitScale` | none / limited / optimal | audience fit |
| `OptionsScale` | none / limited / many | (hyper)personalization options |
| `FivePointScale` | very_low / low / medium / high / very_high | generic 5-point |
| `LengthScale` | right / too_long | sentence / paragraph length |
| `LanguageLevelScale` | A1–C2 | CEFR reading level |
| `UsedScale` | used / not_used | source-ID/version-tag presence (Tracability) |
| `ApplicableScale` | applicable / not_applicable | applicability flags |
| `ExclusionScale` | exclusion / no_exclusion | GenAI source exclusion tag |
| `CESScale` | 1–5 customer-effort | Engagement (out of scope today) |
| `CSATScale` | 1–5 customer-satisfaction | Engagement |
| `NPSScale` | detractor / passive / promoter | Engagement |
| `PublishedScale` | published / not_published | dashboard state |
| `SentimentScale` | neutral / positive / negative | sentiment |
| `GenderScale` | neutral / male / female | inclusivity sub-check |

Each scale also carries a sentinel `unknown = "unknown"` value so an LLM
judge that fails to commit to a real value can be detected (and the gate
fail accordingly).

`PASSING_VALUES: dict[type[Enum], set[Enum]]` defines, per scale, *which*
enum values count as passing the norm. The default `is_passing(scale, value)`
function uses this map; KPIs with bespoke norm semantics override in their
checker.

### 5.2 Output shapes

```python
class KPIResult(BaseModel):
    kpi_id: str
    name: str
    cluster: Optional[str]
    category: Optional[str]
    weight: Literal["Blocking", "High", "Medium", "Low"]
    monitoring: Literal["Mandatory", "Optional"]
    indicator: Optional[str]   # name of the scale enum, e.g. "ErrorScale"
    value: str                 # enum value, or "not_evaluated" / "unknown"
    raw_metric: Optional[dict] # underlying number/snippet for tier 1
    reason: str = ""
    tier: Literal[1, 2, 3]
    passed: bool
    source: Literal["deterministic", "llm", "skipped"] = "deterministic"


class EvaluationResult(BaseModel):
    passed: bool
    failed_blocking: list[str]
    results: list[KPIResult]
    maturity_by_category: dict[str, str]   # category → low/medium/high
    dclp_steps_required: list[str]         # tier-3 KPI ids the workflow must clear
    channel: Channel = "web"
    origin: Origin = "instant"
    model: Optional[str] = None
    source: Literal["deterministic", "llm"] = "deterministic"
    reasoning: str = ""
```

`Channel = Literal["web", "chat", "messages", "employee", "app_ib"]`.
`Origin = Literal["human", "genai_knowledge", "instant"]`.

---

## 6. Catalogue accessor

`backend/evaluation/catalogue.py` exposes a typed view over the JSON:

```python
@dataclass(frozen=True)
class KPI:
    id: str
    name: str
    category: Optional[str]
    cluster_short: Optional[str]
    primary_cluster: Optional[str]
    secondary_cluster: Optional[str]
    monitoring: str               # "Mandatory" | "Optional"
    weight: str                   # "Blocking" | "High" | "Medium" | "Low"
    contribution: Optional[str]
    norm: Optional[str]
    norm_chat: Optional[str]
    norm_messages: Optional[str]
    measurement: Optional[str]
    indicator: Optional[str]      # enum class name
    indicator_phrase: Optional[str]
    automated_match: Optional[str]
    relevance: dict[str, Optional[str]]
    guardrail_category: Optional[str]
    guardrail_tag: Optional[str]

    @property
    def is_blocking(self) -> bool: ...
    def norm_for(self, channel: Channel) -> Optional[str]: ...


@dataclass(frozen=True)
class Catalogue:
    kpis: tuple[KPI, ...]
    categories: tuple[dict, ...]
    clusters: tuple[dict, ...]

    def by_id(self, kpi_id: str) -> KPI: ...
    def blocking(self) -> tuple[KPI, ...]
    def by_weight(self, weight: str) -> tuple[KPI, ...]
    def applicable(self, *, origin: Origin, channel: Channel) -> tuple[KPI, ...]
    def by_category(self) -> dict[str, list[KPI]]


@lru_cache(maxsize=1)
def load_catalogue(path: Optional[Path] = None) -> Catalogue: ...
```

`Catalogue.applicable(...)` is the applicability filter: a KPI applies when
both its origin and channel relevance values start with `Applicable` (or
equal `Need`). Missing values are treated as permissively applicable — the
workbook leaves the per-channel cell blank when a row applies to every
channel.

---

## 7. Tier 1 — deterministic checks

`backend/evaluation/tier1_deterministic.py` ships **11 pure-Python checks**,
each bound to a real catalogue `kpi_id` so the dispatcher in `run_tier1`
walks the applicable subset and ignores everything else.

| KPI id | Function | What it checks |
|---|---|---|
| `sentence_number_of_words` | `check_sentence_length` | ≤ 15 words per sentence (B1), ≤ 10% long-sentence ratio. |
| `paragraph_bubble_number_of_words_sentences` | `check_paragraph_length` | ≤ 100 words per paragraph. |
| `sentence_structure` | `check_passive_voice` | Best-effort passive-voice ratio (language-aware NL vs EN regex). |
| `reading_level` | `check_reading_level_b1` | Flesch reading ease → CEFR band (A1-B1 pass). |
| `bullet_list_points` | `check_bullet_list_presence` | At least one markdown bullet list. |
| `images_with_missing_alt_text` | `check_images_alt_present` | Every `![...]()` has non-empty alt. |
| `h1_header_presence` | `check_h1_count` | Exactly one H1. |
| `h1_header_keywords` | `check_keyword_in_h1` | Top topic keyword present in H1. |
| `tracability` (Blocking) | `check_tracability` | `gen.citations` non-empty (source-ID + version stand-in). |
| `approved_source_content_for_genai` (Blocking) | `check_approved_source_for_genai` | No cited snippet flagged `exclude_for_genai`. |
| `factuality_truthfullness` (Blocking) | `check_factuality_no_hallucinated_citations` | Every `[n]` marker in body maps to a real snippet. Catches obvious hallucination before Tier 2; full judge is in Tier 2. |

### Shared helpers

- `_strip_md` — naive markdown→plaintext.
- `_split_sentences` / `_split_paragraphs` / `_word_count`.
- `_H1_RE`, `_H2_RE`, `_BULLET_RE`, `_LINK_RE`, `_IMAGE_RE`.
- `_count_syllables` — for the Flesch implementation.
- `_result(kpi, …)` — single helper that constructs a `KPIResult` with
  `tier=1`, `source="deterministic"`, and the right indicator name.

Each check is defensive: any exception in the checker is caught by the
dispatcher and emitted as a `KPIResult` with `value="unknown"`, `passed=False`,
`reason="check error: <message>"`. A broken check never crashes the pipeline.

### Wordlist-driven KPIs (not yet wired)

Three KPIs need ABN-AMRO-specific word lists that aren't shipped with the
workbook: `Jeukwoorden`, `BlackList`, `MandatoryList`. The directory
`backend/evaluation/data/wordlists/` is reserved for them; checkers can be
added once the lists land.

---

## 8. Tier 2 — LLM-as-judge

`backend/evaluation/tier2_judges.py` runs **12 rubrics in parallel** via a
`ThreadPoolExecutor(max_workers=6)`. Each judge:

1. Looks up its KPI in the catalogue (binding by `kpi_id`).
2. Builds a tiny per-scale `JudgeOutput(value: scale, reason: str)`
   Pydantic model so the OpenAI `response_format` enforces the enum at
   the schema level — the model literally cannot return a value outside
   the scale.
3. Calls `client.beta.chat.completions.parse(...)` with the shared
   `JUDGE_SYSTEM_PROMPT` and a per-rubric anchor.
4. Returns a `KPIResult` with `tier=2`, `source="llm"`, and `passed`
   derived from `is_passing(scale, value)`.

### The 12 judges (`JUDGES` in `tier2_judges.py`)

| Rubric | KPI id | Scale | Default pass |
|---|---|---|---|
| `factuality` (Blocking) | `factuality_truthfullness` | `ErrorScale` | `few`, `none` |
| `truthfullness` (Blocking) | `truthfullness` | `DeviationScale` | `none` |
| `relevancy` (Blocking) | `relevancy` | `RelevanceScale` | `reasonable`, `highly` |
| `privacy_security` (Blocking) | `privacy_and_security` | `DeviationScale` | `none` |
| `groundedness` (High) | `groundedness_source` | `GroundednessScale` | `reasonable`, `full` |
| `completeness_source` (High) | `completeness_source` | `CompletenessScale` | `mostly`, `full` |
| `comprehensiveness` (High, AEO) | `comprehensiveness_answer` | `CompletenessScale` | `mostly`, `full` |
| `clarity` (High) | `clarity` | `ClarityScale` | `clear`, `very_clear` |
| `uniqueness_added_value` (GenAI rater) | `body_content_uniqueness` | `PresenceScale` | `present` |
| `demonstrable_expertise` (GenAI rater) | `experience_expertise` | `PresenceScale` | `present` |
| `no_paraphrase` (GenAI rater) | `no_paraphrase` (synthetic) | `PresenceScale` | `present` |
| `no_filler` (GenAI rater) | `no_filler` (synthetic) | `PresenceScale` | `present` |

### Prompts (`prompt.py`)

- One shared `JUDGE_SYSTEM_PROMPT` describing the judge persona, the rubric/
  evidence/draft inputs, and the indicator-scale constraint.
- A `RUBRICS: dict[str, str]` mapping each rubric name to a short anchor
  (\~3 sentences) that explains what the dimension means in concrete terms.
- `build_judge_user_message(rubric_name, req, gen, allowed_values)` —
  assembles the user-role message with rubric + query + snippets + draft +
  allowed values listed prose-style (belt-and-braces alongside the schema-
  level constraint).

### Failure handling

- **Stub path** (no api_key or no model): every judge emits a `skipped`
  `KPIResult` with `value="not_evaluated"`, `passed=True` (permissive
  default), `source="skipped"`. The envelope still lists every KPI so the
  UI can render the full taxonomy.
- **Individual judge failure**: caught at the `as_completed()` level, the
  KPI is emitted with `value="unknown"`, `passed=False`,
  `reason="judge error: <message>"`, and the other 11 judges proceed
  unaffected.
- **Stable ordering**: results are sorted by spec registration order
  after parallel completion so the UI renders deterministically across
  reruns.

### Cost shape

- 12 judges × \~300 tokens of context = \~3.6K tokens per evaluation, in
  parallel.
- Default model: `gpt-4o-mini` (set in the frontend), which is more than
  enough for one-dimension scoring.

---

## 9. Tier 3 — dCLP human-loop flags

`backend/evaluation/tier3_human_loop.py` declares which dCLP signoff steps
are required for the current (`origin`, `channel`); it does not attempt to
execute them. The four IDs are constant:

```python
DCLP_STEP_IDS = (
    "human_expert_check_substance",
    "human_expert_check_compliancy_legal",
    "human_expert_check_content",
    "status_of_evaluation",
)
```

The service emits one `KPIResult` per applicable step with `tier=3`,
`value="no"`, `passed=False`, `source="deterministic"`. The workflow system
records the actual signoff outside AURORA; when it lands, a future revision
can overlay the recorded state onto these results.

---

## 10. Service flow

`backend/evaluation/service.py`:

```python
def evaluate(
    req: ContentRequest,
    gen: ContentResult,
    *,
    channel: Channel = "web",
    origin: Origin = "instant",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    strict_mode: bool = False,
    catalogue: Optional[Catalogue] = None,
) -> EvaluationResult:
```

Sequence:

1. **Load catalogue** (lru-cached).
2. **Run Tier 1**. Collect results.
3. **Short-circuit gate**: if any Mandatory + Blocking KPI failed in
   Tier 1, append Tier 3 pending entries and return immediately with
   `passed=False`. Tier 2 is **not run** — no point burning LLM spend on
   content that has already failed an obvious deterministic gate.
4. **Run Tier 2** (LLM mode or skipped stubs).
5. **Strict-mode flip**: if `strict_mode=True` and `source="skipped"`,
   override `passed=False` so dev-mode-without-keys cannot approve content.
6. **Tier 3**: append `pending_results(...)` for the applicable dCLP steps.
7. **Aggregate**:
   - `failed_blocking` — `kpi_id`s of failing Mandatory + Blocking KPIs
     across all tiers.
   - `maturity_by_category` — per-category `passed / total` ratio mapped
     to `low` (< 0.5), `medium` (< 0.8), `high` (≥ 0.8). Tier 3 entries
     are excluded from this rollup (they are workflow state, not a quality
     verdict).
   - `dclp_steps_required` — the Tier 3 IDs.
8. **Return** with `source="llm"` if Tier 2 ran, `"deterministic"` otherwise.
9. **Never raise**: every failure mode produces a result envelope so the
   UI can render even when half the judges errored.

---

## 11. Synthetic KPIs

Two GenAI-rater rubrics from the Criteria sheet (Cols T/U) do not have rows
in the Inventory sheet:

- **No paraphrased content** → `no_paraphrase`
- **No filler / too generic information** → `no_filler`

The build script materialises them as Optional + High KPIs under
*Online findability and visibility / content value* with
`indicator: PresenceScale`. Both are flagged `source_block: "Criteria sheet —
Search quality raters for GenAI"` in the catalogue for traceability. These
are the only places the implementation extends the workbook rather than
mirroring it.

---

## 12. Frontend integration

Two files touched: `frontend/app.py` and `frontend/pages/2_Settings.py`.

### 12.1 `frontend/app.py`

| Edit | Detail |
|---|---|
| Imports | `from backend.evaluation import EvaluationResult, KPIResult, evaluate` |
| `_STAGE_STYLES` | Added `evaluation` entry — avatar 🛡️, cyan tint `#cffafe` / `#155e75`, label "Stage 6 · Evaluation". |
| Session-state seeds | `eval_api_key` (from env `OPENAI_API_KEY_EVALUATION`), `eval_model` (default `gpt-4o-mini`), `eval_strict_mode` (False). |
| Stage CSS | Added `evaluation` to the shared selector + a dedicated tint rule mirroring the other stages. |
| `_proceed_to_generation` | After appending the `content` message, auto-runs `evaluate(req, result, channel="web", origin="instant", api_key=..., model=..., strict_mode=...)` and appends an `evaluation` message. Spinner caption differs in stub vs LLM mode. |
| `_render_evaluation_message` | The renderer (see below). |
| `_TIMELINE_LABELS` | Added `evaluation: "Evaluation"`. |
| Scroll-spy `COLOR_BY_KIND` | Added `evaluation: '#155e75'`. |
| Replay loop | `elif m.get("kind") == "evaluation": _render_evaluation_message(idx, m)`. |

### 12.2 What `_render_evaluation_message` shows

Inside one cyan-tinted assistant bubble (Stage 6 chip + anchor):

1. **Caption line** — model + channel + origin (or stub-mode warning).
2. **Verdict banner** — green `✅ Passed — no blocking KPI violations
   detected.` or red `⛔ Blocked — N blocking KPI failure(s): kpi_id, ...`.
3. **Maturity by category** — one bullet per category present in
   `maturity_by_category`, with 🟢/🟡/🔴 colour-coded dots.
4. **Editorial signoff required (dCLP)** — bulleted list of the
   `dclp_steps_required` entries with ⏳ icons (only when origin =
   `genai_knowledge`).
5. **Detailed KPI breakdown** — collapsible expander showing each KPI grouped
   by tier (Tier 1 / Tier 2 / Tier 3), weight-sorted Blocking → High →
   Medium → Low, with failures floated above passes. Per row:
   `✅ / ⏭️ / ❌  [BLOCKING chip if applicable]  **Name**  · indicator=… · value=… — reason`.

### 12.3 `frontend/pages/2_Settings.py`

| Edit | Detail |
|---|---|
| Session-state seeds | Same three keys as `app.py`. |
| New `Evaluation (Stage 6)` section | Text inputs for **Evaluation API Key** + **Evaluation Model**, plus a **Strict mode** toggle. |
| Save handler | Reads + strips the three new fields and writes them back to session state. |
| Status footer | New info/warning/error banner mirroring the content-generation pattern. With strict mode ON *and* no key, prints an error (because Tier 2 KPIs would then auto-fail). |

---

## 13. Sample usage

### From Python

```python
from backend.evaluation import evaluate

result = evaluate(
    req,              # ContentRequest from Stage 5
    gen,              # ContentResult from Stage 5
    channel="web",
    origin="instant",
    api_key=os.getenv("OPENAI_API_KEY_EVALUATION"),
    model="gpt-4o-mini",
    strict_mode=False,
)

result.passed                  # bool — gate decision
result.failed_blocking         # list[str]
result.results                 # list[KPIResult] across all 3 tiers
result.maturity_by_category    # {"Compliancy & substantive quality": "high", ...}
result.dclp_steps_required     # list[str] for the workflow system
```

### Result shape (real run, stub mode)

```text
passed=True, blocking=[], source=deterministic
maturity: {
  "Accessibility & inclusion":   "high",
  "Compliancy  & substantive quality": "high",
  "Online findability & visibility":   "high",
  "Online findability and visibility": "high",
}
counts: t1=9, t2=12 (skipped), t3=0
```

### Result shape (Blocking short-circuit)

```text
passed=False, blocking=['factuality_truthfullness']
source=deterministic
reasoning="Tier 1 short-circuit on blocking KPI failure"
counts: t1=N, t2=0, t3=0   # Tier 2 skipped
```

---

## 14. Tests

`backend/evaluation/tests/` — 28 tests, all passing under
`pytest backend/`.

- **`test_tier1.py`** — 19 unit tests, one per check. Crafted minimal
  bodies prove pass/fail behaviour and that the right indicator enum is
  used.
- **`test_tier2.py`** — 3 tests:
  - Stub path: all 12 specs return `source="skipped"`, all pass.
  - LLM path: mocked client routed by scale name; every judge returns a
    passing enum value; all results `passed=True`.
  - Isolated failure: one judge call raises; that KPI is `value="unknown"`,
    `passed=False`; the other 11 continue.
- **`test_service.py`** — 5 tests:
  - Stub-path happy path.
  - Tier 1 Blocking short-circuit (Tier 2 not run at all).
  - GenAI-knowledge origin lists all 4 dCLP steps.
  - `strict_mode=True` flips skipped Tier 2 entries to failing.
  - Result envelope carries channel/origin/model/source metadata.
- **`fixtures.py`** — `make_request()` / `make_generation()` for parity
  with the other stage test files.

Full backend suite still passes (57 tests).

---

## 15. Design choices flagged

These were live decisions during the implementation. None are load-bearing
on the code — easy follow-ups if you decide differently.

- **`origin="instant"` hard-coded in `_proceed_to_generation`.** AURORA's
  current pipeline retrieves snippets via PageIndex, so `genai_knowledge`
  is the strictly more accurate origin — but then every eval blocks on the
  4 dCLP signoff KPIs by default, which is noisy for a dev UI. A Settings
  dropdown ("Content origin: instant / genai_knowledge / human") is the
  natural next step.
- **`channel="web"` hard-coded.** Fine for the editorial co-pilot today;
  add a per-prompt control when AURORA targets chat / messages content.
- **Strict mode default OFF.** Dev runs aren't blocked by missing keys.
  Flip to `True` for production.
- **No backwards-compatibility shim for `Snippet.exclude_for_genai`.**
  The check reads `getattr(snip, "exclude_for_genai", False)` — when the
  corpus pipeline starts emitting that field, the eval picks it up
  automatically; nothing breaks today.
- **Catalogue rebuild is manual.** A CI assertion that the on-disk JSON
  matches `xlsx_hash` is the natural follow-up.

---

## 16. What is **not** done

- Textmetrics API integration. We run regex-style equivalents locally; if
  the bank prefers vendor-canonical results, swap in the Textmetrics call
  inside the existing Tier 1 functions.
- Engagement KPIs (CSAT / CES / NPS / clicks). Post-publish telemetry, not
  generation-time eval.
- The 4 dCLP human-signoff *workflow* (the UI flags them; recording is
  outside AURORA).
- PowerBI write-back. The evaluator emits the indicator-enum values
  defined by the workbook, so it's a thin shim to add later.
- Wordlist-driven Tier 1 checks (Jeukwoorden, BlackList, MandatoryList).
  Slot reserved in `backend/evaluation/data/wordlists/`; checkers can be
  added once the lists land.
- Channel-specific evaluation flows beyond `web`. Catalogue carries
  channel-norm overrides; only the frontend caller is web-only.
