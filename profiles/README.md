# `profiles/` — AURORA profile registry

Machine-readable form of the profiles described in
[`docs/profiles.md`](../docs/profiles.md). Two axes — **workflow** (how the
user is working) and **domain expert** (what the work is about) — one YAML
file per profile.

## Layout

```
profiles/
├── _schema.md              # human-readable schema reference
├── loader.py               # dataclasses + load_all / load_by_id / match
├── validate.py             # `python -m profiles.validate`
├── workflow/
│   ├── drafter.yaml
│   ├── reviewer.yaml
│   └── curator.yaml
├── domain_expert/
│   ├── julia_krauwer.yaml
│   ├── mario_bersem.yaml
│   └── amad_khan.yaml
└── tests/test_profiles.py
```

Files prefixed with `_` are not loaded as profiles.

## Usage

```python
import profiles

# Every profile, parsed and validated
bundle = profiles.load_all()

# Lookup by id
drafter = profiles.load_by_id("drafter")

# Activation (see docs/profiles.md §3)
matched = profiles.match(
    intent_code="T1_DRAFT",
    sector="Technologie, Media & Telecom",
    keywords=["cybersecurity"],
)
# → matched.workflow      = (drafter,)
# → matched.domain_expert = (expert_julia_krauwer,)
```

## Validation

```bash
python -m profiles.validate
```

Checks every YAML parses, that `co_activates_with` references resolve, and
that workflow `intent_codes` are in the canonical set
(`T1_DRAFT`, `T1_TRANSLATE`, `T1_SEARCH`, `T2_COMPLIANCE`, `T4_RENEWAL`).

## Adding a new profile

1. Pick the axis: `workflow/` or `domain_expert/`.
2. Copy the closest existing YAML and edit. Required fields are listed in
   [`_schema.md`](_schema.md).
3. Run `python -m profiles.validate` to confirm it parses and references
   resolve.
4. Update [`docs/profiles.md`](../docs/profiles.md) — the doc is the
   narrative source of truth; YAMLs are the machine source.

## Status

Six profiles ship with this registry — three workflow (Drafter, Reviewer,
Curator) and three TMT domain experts (Julia Krauwer, Mario Bersem, Amad
Khan). The TMT experts are sourced from the local NL/EN article corpus and
need confirmation from the authors; see
[`docs/profiles.md`](../docs/profiles.md) §8 for the open TODO list.
