# AURORA Evaluation

The evaluation stage is based on ABN AMRO's `Content KPI inventory_AISO.xlsx`
workbook, exported to `assets/evaluation/kpi_catalogue.json` by
`archive/rag/scripts/build_kpi_catalogue.py`.

The workbook is broader than the generation step. It contains publication,
SEO, accessibility, lifecycle, PowerBI, Textmetrics, and dCLP signoff KPIs.
AURORA therefore treats the workbook as the source catalogue, but the default
generation gate uses only checks that are necessary and reliable at draft time.

## Workbook Reading

The workbook's Inventory sheet lists 135 runtime KPI entries in the catalogue:

- 132 leaf KPIs from Inventory rows 9-169.
- 3 synthetic generated-content quality rubrics from the Criteria sheet.
- 10 Mandatory + Blocking leaf KPIs in the runtime catalogue.

The blocking rows are not all suitable as automatic generation blockers:

- `human_expert_check_substance`, `human_expert_check_compliancy_legal`,
  `human_expert_check_content`, and `status_of_evaluation` are dCLP or
  lifecycle workflow states. AURORA reports them as pending signoffs where
  applicable, but they do not decide whether a draft is usable.
- `tracability` is scoped by the workbook to GenAI source content for instant
  output. It blocks `genai_knowledge` source content when source IDs are
  missing, but it does not reject a normal instant draft only because citations
  are absent.
- `approved_source_content_for_genai` is objective and remains a hard stop when
  a cited snippet is explicitly tagged `exclude_for_genai`.
- `factuality_truthfullness`, `truthfullness`, `relevancy`, and
  `privacy_and_security` are judgemental content checks. They run as LLM judges
  when configured, with softened draft-stage thresholds.

## Default Gate

Default mode blocks only material generation problems:

| KPI | Blocks when |
|---|---|
| `factuality_truthfullness` | deterministic citation markers are invalid, or the judge returns `moderate`, `several`, or `numerous` errors |
| `relevancy` | the judge returns `off_topic` |
| `truthfullness` | the judge returns `many` deviations |
| `privacy_and_security` | the judge returns `many` deviations |
| `approved_source_content_for_genai` | a cited source is explicitly excluded for GenAI use |
| `tracability` | `origin=genai_knowledge` and no source citations are present |

Minor judge findings such as `few errors`, `few deviations`, or `somewhat`
relevant remain visible in the KPI result, but they do not fail the generation
stage by default.

`strict_mode=True` restores fail-closed behavior for Mandatory + Blocking
Tier 1 and Tier 2 checks. Use strict mode for production approval workflows
that deliberately want missing LLM judges, judge errors, or minor blocking-KPI
deviations to stop the process.

## Active Checks

Tier 1 now runs a small deterministic set:

- `factuality_truthfullness`: catches impossible citation markers such as
  `[99]` when only two snippets were retrieved.
- `images_with_missing_alt_text`: reports missing image alt text when markdown
  images are present.
- `tracability`: runs where the workbook makes it applicable, especially
  `genai_knowledge` source content.
- `approved_source_content_for_genai`: runs when a draft cites sources, so
  explicit exclusion metadata is enforced.

Tier 2 runs six essential judges:

- `factuality_truthfullness`
- `truthfullness`
- `relevancy`
- `privacy_and_security`
- `groundedness_source`
- `comprehensiveness_answer`

The previous SEO, readability, passive voice, heading, keyword, referral,
clarity, CEFR, uniqueness, no-paraphrase, no-filler, and expertise checks are
not part of the default generation gate. Their checker functions and catalogue
rows remain available for a future stricter audit profile.

## Tier 3

Tier 3 emits pending dCLP signoff results for applicable workflow KPIs. These
results are intentionally not blockers in default mode because they are not
content-generation verdicts and cannot be completed by AURORA itself.
