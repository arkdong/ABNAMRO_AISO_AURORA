"""Top-level evaluate() — orchestration tests.

Covers:
1. Stub path (no LLM): runs Tier 1 + skipped Tier 2, passes the gate.
2. Blocking short-circuit: a Tier 1 Blocking failure skips Tier 2 entirely.
3. dCLP requirements appear for ``genai_knowledge`` origin.
4. ``strict_mode`` flips skipped Tier 2 to failing.
"""

from __future__ import annotations

from backend.content_generation.types import Citation, ContentResult
from backend.evaluation import evaluate
from backend.evaluation.tests.fixtures import make_generation, make_request


def test_stub_path_passes_with_good_content():
    req, gen = make_request(), make_generation()
    r = evaluate(req, gen, api_key=None, model=None)
    assert r.source == "deterministic"
    assert r.passed
    assert r.failed_blocking == []
    # All Tier 2 entries are skipped.
    tier2 = [k for k in r.results if k.tier == 2]
    assert tier2
    assert all(k.source == "skipped" for k in tier2)
    # Maturity rollup is populated.
    assert r.maturity_by_category


def test_blocking_short_circuits_when_tier1_fails():
    req = make_request()
    # Hallucinated citation index ([99]) fails the factuality floor.
    gen = ContentResult(
        body="# Title\n\nClaim [99] without backing.",
        citations=[],
        model=None,
        source="llm",
    )
    r = evaluate(req, gen, api_key=None, model=None)
    assert not r.passed
    assert "factuality_truthfullness" in r.failed_blocking
    # Tier 2 was skipped (short-circuit) — no Tier 2 entries at all.
    tier2 = [k for k in r.results if k.tier == 2]
    assert tier2 == []


def test_blocking_tracability_fails_for_genai_source():
    # For ``genai_knowledge`` origin the tracability KPI applies; empty
    # citations make it fail and block publication.
    req = make_request()
    gen = ContentResult(body="# Title\n\nBody without citations.", citations=[])
    r = evaluate(req, gen, origin="genai_knowledge", api_key=None, model=None)
    assert not r.passed
    assert "tracability" in r.failed_blocking


def test_genai_knowledge_origin_lists_dclp_steps():
    req, gen = make_request(), make_generation()
    r = evaluate(req, gen, origin="genai_knowledge", api_key=None, model=None)
    # The four dCLP-step KPIs should be required for genai_knowledge content.
    assert {
        "human_expert_check_substance",
        "human_expert_check_compliancy_legal",
        "human_expert_check_content",
        "status_of_evaluation",
    }.issubset(set(r.dclp_steps_required))
    # And they appear as tier-3 pending results.
    tier3 = [k for k in r.results if k.tier == 3]
    assert {k.kpi_id for k in tier3} == set(r.dclp_steps_required)
    assert all(not k.passed for k in tier3)  # pending == not_yet_signed_off


def test_strict_mode_flips_skipped_to_failing():
    req, gen = make_request(), make_generation()
    r = evaluate(req, gen, api_key=None, model=None, strict_mode=True)
    tier2 = [k for k in r.results if k.tier == 2]
    assert tier2
    # In strict mode, all skipped tier-2 entries are marked failing.
    assert all(not k.passed for k in tier2)
    # None of them are Blocking, so the gate still passes (Blocking failures
    # would have appeared in failed_blocking).
    blocking_failures = [
        k for k in tier2 if k.weight == "Blocking" and not k.passed
    ]
    if blocking_failures:
        assert all(b.kpi_id in r.failed_blocking for b in blocking_failures)


def test_evaluation_result_envelope_metadata():
    req, gen = make_request(), make_generation()
    r = evaluate(req, gen, channel="chat", origin="human", api_key=None, model=None)
    assert r.channel == "chat"
    assert r.origin == "human"
    assert r.model is None
    # Source is deterministic when no API key is configured.
    assert r.source == "deterministic"
