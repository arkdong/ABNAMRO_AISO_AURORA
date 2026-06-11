"""Smoke test for ``build_query``: intent + bundle fields plumb through."""

from __future__ import annotations

from backend.intent import IntentResult
from backend.retrieval import build_query
from profiles import DomainExpertProfile, ProfileBundle, WorkflowProfile


def _intent() -> IntentResult:
    return IntentResult(
        role="Insights Editorial",
        task_codes=["T1_DRAFT", "T1_SEARCH"],
        confidence=0.9,
        task_reason="test",
        sector="Technologie, Media & Telecom",
        topic_keywords=["agentic AI", "cybersecurity"],
        language="en",
    )


def _bundle() -> ProfileBundle:
    w = WorkflowProfile(
        id="drafter",
        name="Drafter",
        description="x",
        activates_on_intent_codes=("T1_DRAFT",),
        knowledge=(),
        skills=(),
        tools=(),
        guardrails=(),
        outputs=(),
        co_activates_with=(),
    )
    e = DomainExpertProfile(
        id="expert_tmt_cybersecurity",
        name="X",
        description="x",
        sector="Technologie, Media & Telecom",
        topic_keywords=(),
        knowledge=(),
        expertise_areas=(),
        style_signature=(),
        co_activates_with=(),
    )
    return ProfileBundle(workflow=(w,), domain_expert=(e,))


def test_build_query_plumbs_all_fields():
    q = build_query("the prompt", _intent(), _bundle(), k=7)
    assert q.user_prompt == "the prompt"
    assert q.task_codes == ["T1_DRAFT", "T1_SEARCH"]
    assert q.sector == "Technologie, Media & Telecom"
    assert q.topic_keywords == ["agentic AI", "cybersecurity"]
    assert q.language == "en"
    assert q.workflow_profile_ids == ["drafter"]
    assert q.expert_profile_ids == ["expert_tmt_cybersecurity"]
    assert q.k == 7
