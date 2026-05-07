"""Validate every profile YAML.

Run as a script (`python -m profiles.validate`) or import `validate()` for tests.
Checks that every profile parses, references in `co_activates_with` resolve,
and every workflow `intent_code` is in the canonical set.
"""

from __future__ import annotations

import sys
from typing import Iterable

from .loader import (
    CANONICAL_INTENT_CODES,
    DomainExpertProfile,
    WorkflowProfile,
    load_all,
)


def validate() -> list[str]:
    """Return a list of error strings. Empty list = all profiles valid."""
    errors: list[str] = []
    bundle = load_all()
    all_ids = {p.id for p in bundle}

    seen: set[str] = set()
    for p in bundle:
        if p.id in seen:
            errors.append(f"duplicate profile id: {p.id}")
        seen.add(p.id)

        for ref in p.co_activates_with:
            if ref not in all_ids:
                errors.append(f"{p.id}: co_activates_with references unknown profile '{ref}'")

        if isinstance(p, WorkflowProfile):
            if not p.activates_on_intent_codes:
                errors.append(f"{p.id}: workflow profile has no intent_codes")
            for code in p.activates_on_intent_codes:
                if code not in CANONICAL_INTENT_CODES:
                    errors.append(
                        f"{p.id}: intent_code '{code}' not in canonical set "
                        f"{sorted(CANONICAL_INTENT_CODES)}"
                    )

        elif isinstance(p, DomainExpertProfile):
            if not p.sector:
                errors.append(f"{p.id}: domain_expert missing sector")
            if not p.topic_keywords:
                errors.append(f"{p.id}: domain_expert has no topic_keywords")

    return errors


def _print_errors(errors: Iterable[str]) -> None:
    for e in errors:
        print(f"  ✗ {e}", file=sys.stderr)


def main() -> int:
    errors = validate()
    if errors:
        print(f"profile validation FAILED ({len(errors)} error(s)):", file=sys.stderr)
        _print_errors(errors)
        return 1
    bundle = load_all()
    total = len(bundle.workflow) + len(bundle.domain_expert)
    print(
        f"profile validation OK — {total} profiles "
        f"({len(bundle.workflow)} workflow, {len(bundle.domain_expert)} domain expert)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
