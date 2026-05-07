# Profile schema

Mirrors `docs/profiles.md` §2. The dataclasses in `profiles/loader.py` are the
executable source of truth; this file is for humans editing or reviewing YAMLs.

## Common fields (every profile)

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable identifier, snake_case (e.g. `drafter`, `expert_julia_krauwer`). |
| `name` | string | Human-readable. |
| `description` | string | One-line summary. |
| `category` | `workflow` \| `domain_expert` | Which axis. |
| `activates_on` | object | Selection rules — shape depends on category (see below). |
| `knowledge` | string[] | Documents / data sources consulted. |
| `capabilities` | object | What the profile knows how to do — shape depends on category. |
| `co_activates_with` | string[] | IDs of commonly paired profiles. |

## Workflow-only fields

| Field | Type | Notes |
|---|---|---|
| `activates_on.intent_codes` | string[] | e.g. `[T1_DRAFT, T1_TRANSLATE]`. |
| `capabilities.skills` | string[] | Verbs (draft, score, search). |
| `tools` | string[] | Concrete callable tools. |
| `guardrails` | string[] | Hard rules. |
| `outputs` | string[] | What this profile emits. |

## Domain-expert-only fields

| Field | Type | Notes |
|---|---|---|
| `activates_on.sector` | string | e.g. `"Technologie, Media & Telecom"`. |
| `activates_on.topic_keywords` | string[] | Tags / keywords this expert owns. |
| `capabilities.expertise_areas` | string[] | Nouns (5G, AI security, retail media). |
| `style_signature` | string[] | Recurring stylistic patterns. |

## Canonical intent codes

Used in `workflow.activates_on.intent_codes`:

- `T1_DRAFT` — draft new content
- `T1_TRANSLATE` — translate existing content
- `T1_SEARCH` — search corpus for related articles
- `T2_COMPLIANCE` — quality & compliance check
- `T4_RENEWAL` — detect & renew aging articles

## Required fields

Every profile must define: `id`, `name`, `description`, `category`,
`activates_on`, `knowledge`, `capabilities`, `co_activates_with`.

Files prefixed with `_` (this one) are not loaded as profiles.
