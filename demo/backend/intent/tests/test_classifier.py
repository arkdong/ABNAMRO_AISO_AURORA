"""Deterministic-fallback enrichment tests for backend.intent.classifier.

Each test runs ``classify_full`` with no API key (forces the deterministic
path), then feeds the resulting fields into ``profiles.match()`` and asserts
the profiles surfaced match the demo doc's expectation table.
"""

from __future__ import annotations

from backend.intent import classify_full
from profiles import match

QUERY_1 = (
    "Write a short analysis article in English on how Agentic AI is changing "
    "the cybersecurity arms race for Dutch TMT companies, and what the "
    "workforce-shortage angle means for IT-leveranciers."
)

QUERY_2 = (
    "Check this draft about retail media and the Digital Services Act against "
    "our writing guide — flag anything that breaks a hard rule."
)

QUERY_3 = (
    "Vertaal het artikel 'The two faces of Agentic AI' naar het Nederlands en "
    "laat me zien welke gerelateerde artikelen we al hebben."
)


def test_query1_routes_to_drafter_plus_cybersecurity_expert():
    result, source = classify_full(QUERY_1, api_key=None, model=None)
    assert source == "deterministic"
    assert result.task_codes == ["T1_DRAFT"]
    assert result.sector == "Technologie, Media & Telecom"
    assert result.language == "en"
    # cybersecurity expert keywords picked up
    kw_lower = {k.lower() for k in result.topic_keywords}
    assert "agentic ai" in kw_lower
    assert "cybersecurity" in kw_lower

    bundle = match(
        intent_code=result.task_codes[0],
        sector=result.sector,
        keywords=result.topic_keywords,
    )
    assert {p.id for p in bundle.workflow} == {"drafter"}
    assert "expert_tmt_cybersecurity" in {p.id for p in bundle.domain_expert}


def test_query2_routes_to_reviewer_plus_media_expert():
    result, source = classify_full(QUERY_2, api_key=None, model=None)
    assert source == "deterministic"
    assert result.task_codes == ["T2_COMPLIANCE"]
    assert result.sector == "Technologie, Media & Telecom"
    kw_lower = {k.lower() for k in result.topic_keywords}
    assert "retail media" in kw_lower
    assert "digital services act" in kw_lower or "dsa" in kw_lower

    bundle = match(
        intent_code=result.task_codes[0],
        sector=result.sector,
        keywords=result.topic_keywords,
    )
    assert {p.id for p in bundle.workflow} == {"reviewer"}
    assert "expert_tmt_media_advertising" in {p.id for p in bundle.domain_expert}


def test_query3_translate_routes_to_drafter_in_dutch():
    # Deterministic fallback is single-intent; demo doc explicitly notes
    # multi-intent only fires on the LLM path. We assert the primary code +
    # NL language detection still route correctly.
    result, source = classify_full(QUERY_3, api_key=None, model=None)
    assert source == "deterministic"
    assert result.task_codes == ["T1_TRANSLATE"]
    assert result.sector == "Technologie, Media & Telecom"
    assert result.language == "nl"
    kw_lower = {k.lower() for k in result.topic_keywords}
    assert "agentic ai" in kw_lower

    bundle = match(
        intent_code=result.task_codes[0],
        sector=result.sector,
        keywords=result.topic_keywords,
    )
    assert "drafter" in {p.id for p in bundle.workflow}  # T1_TRANSLATE → drafter
    assert "expert_tmt_cybersecurity" in {p.id for p in bundle.domain_expert}


def test_legacy_classify_tuple_shape_preserved():
    from backend.intent import classify

    role, task_code, confidence, reason, raw, source = classify(
        QUERY_1, api_key=None, model=None
    )
    assert task_code == "T1_DRAFT"
    assert role
    assert 0.0 <= confidence <= 1.0
    assert reason
    assert raw is None  # deterministic path emits no raw JSON
    assert source == "deterministic"
