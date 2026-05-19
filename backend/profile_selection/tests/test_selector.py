"""Tests for backend.profile_selection.select.

Covers the three demo queries plus two edge cases (no-sector, no-keywords).
"""

from __future__ import annotations

from backend.intent import IntentResult
from backend.profile_selection import select


def _intent(
    *,
    task_codes: list[str],
    sector: str | None = None,
    topic_keywords: list[str] | None = None,
) -> IntentResult:
    return IntentResult(
        role="Insights Editorial",
        task_codes=task_codes,
        confidence=0.9,
        task_reason="test",
        sector=sector,
        topic_keywords=topic_keywords or [],
        language=None,
    )


def test_query1_drafter_plus_cybersecurity_expert():
    bundle = select(
        _intent(
            task_codes=["T1_DRAFT"],
            sector="Technologie, Media & Telecom",
            topic_keywords=["agentic AI", "cybersecurity"],
        )
    )
    assert {p.id for p in bundle.workflow} == {"drafter"}
    expert_ids = {p.id for p in bundle.domain_expert}
    assert "expert_tmt_cybersecurity" in expert_ids


def test_query2_reviewer_plus_media_expert():
    bundle = select(
        _intent(
            task_codes=["T2_COMPLIANCE"],
            sector="Technologie, Media & Telecom",
            topic_keywords=["retail media", "Digital Services Act"],
        )
    )
    assert {p.id for p in bundle.workflow} == {"reviewer"}
    assert {p.id for p in bundle.domain_expert} == {"expert_tmt_media_advertising"}


def test_query3_multi_intent_unions_drafter_and_curator():
    bundle = select(
        _intent(
            task_codes=["T1_TRANSLATE", "T1_SEARCH"],
            sector="Technologie, Media & Telecom",
            topic_keywords=["agentic AI"],
        )
    )
    assert {p.id for p in bundle.workflow} == {"drafter", "curator"}
    # Domain experts must be deduped across the two match() calls.
    expert_ids = [p.id for p in bundle.domain_expert]
    assert len(expert_ids) == len(set(expert_ids))
    assert "expert_tmt_cybersecurity" in expert_ids


def test_no_sector_yields_workflow_only():
    bundle = select(_intent(task_codes=["T1_DRAFT"], sector=None))
    assert {p.id for p in bundle.workflow} == {"drafter"}
    assert bundle.domain_expert == ()


def test_sector_without_keywords_returns_all_sector_experts():
    bundle = select(
        _intent(
            task_codes=["T1_DRAFT"],
            sector="Technologie, Media & Telecom",
            topic_keywords=[],
        )
    )
    # With no keyword filter, the loader returns every expert in the sector.
    expert_ids = {p.id for p in bundle.domain_expert}
    assert expert_ids == {
        "expert_tmt_cybersecurity",
        "expert_tmt_generalist",
        "expert_tmt_media_advertising",
    }
