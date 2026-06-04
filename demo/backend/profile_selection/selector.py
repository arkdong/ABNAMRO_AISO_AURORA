"""Profile selection — turn an :class:`IntentResult` into a :class:`ProfileBundle`.

Multi-intent handling: ``profiles.match()`` is called once per task code, and
the results are unioned (workflow profiles deduped by ``id``; domain experts
are sector-driven so they'd be identical across codes, but we dedupe anyway).
"""

from __future__ import annotations

from backend.intent import IntentResult
from profiles import DomainExpertProfile, ProfileBundle, WorkflowProfile, match


def select(intent: IntentResult) -> ProfileBundle:
    """Return profiles activated by ``intent`` across all of its ``task_codes``."""
    workflow_by_id: dict[str, WorkflowProfile] = {}
    expert_by_id: dict[str, DomainExpertProfile] = {}

    for code in intent.task_codes:
        bundle = match(
            intent_code=code,
            sector=intent.sector,
            keywords=intent.topic_keywords,
        )
        for w in bundle.workflow:
            workflow_by_id.setdefault(w.id, w)
        for e in bundle.domain_expert:
            expert_by_id.setdefault(e.id, e)

    return ProfileBundle(
        workflow=tuple(workflow_by_id.values()),
        domain_expert=tuple(expert_by_id.values()),
    )
