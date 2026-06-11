"""Top-level evaluate_draft() — orchestration tests.

Covers:
1. Stub path (no LLM): runs Tier 1 + skipped Tier 2, passes the gate.
2. Material Tier 1 blockers still short-circuit.
3. dCLP requirements appear for ``genai_knowledge`` origin.
4. ``strict_mode`` flips skipped Blocking Tier 2 checks to failing.
"""

from __future__ import annotations

from aurora_tool_server.evaluation import evaluate_draft
from aurora_tool_server.evaluation.service import _gate_blocking
from aurora_tool_server.schemas import ContentResult, KPIResult
from eval_fixtures import make_generation, make_request


def _evaluate(req, gen, **kwargs):
    return evaluate_draft(
        refined_prompt=req.refined_prompt,
        content=gen,
        snippets=req.snippets,
        intent=req.intent,
        **kwargs,
    )


def test_stub_path_passes_with_good_content():
    req, gen = make_request(), make_generation()
    r = _evaluate(req, gen, api_key=None, model=None)
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
    r = _evaluate(req, gen, api_key=None, model=None)
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
    r = _evaluate(req, gen, origin="genai_knowledge", api_key=None, model=None)
    assert not r.passed
    assert "tracability" in r.failed_blocking


def test_genai_knowledge_origin_lists_dclp_steps():
    req, gen = make_request(), make_generation()
    r = _evaluate(req, gen, origin="genai_knowledge", api_key=None, model=None)
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
    r = _evaluate(req, gen, api_key=None, model=None, strict_mode=True)
    tier2 = [k for k in r.results if k.tier == 2]
    assert tier2
    # In strict mode, all skipped tier-2 entries are marked failing.
    assert all(not k.passed for k in tier2)
    assert not r.passed
    blocking_failures = [k for k in tier2 if k.weight == "Blocking"]
    assert blocking_failures
    assert all(b.kpi_id in r.failed_blocking for b in blocking_failures)


def test_evaluation_result_envelope_metadata():
    req, gen = make_request(), make_generation()
    r = _evaluate(req, gen, channel="chat", origin="human", api_key=None, model=None)
    assert r.channel == "chat"
    assert r.origin == "human"
    assert r.model is None
    # Source is deterministic when no API key is configured.
    assert r.source == "deterministic"


def test_tier3_pending_signoffs_do_not_gate():
    # Workbook dCLP steps are workflow signoff flags, not content verdicts —
    # they are reported (dclp_steps_required + tier-3 results) but must not
    # make every human/genai_knowledge evaluation structurally fail.
    req, gen = make_request(), make_generation()
    r = _evaluate(req, gen, origin="genai_knowledge", api_key=None, model=None)
    assert r.dclp_steps_required  # signoffs still surfaced
    assert r.passed  # ...but pending signoffs don't fail the gate
    assert all(k not in r.failed_blocking for k in r.dclp_steps_required)


def test_tracability_does_not_block_instant_origin():
    # The workbook scopes tracability to GenAI source content for instant
    # output. Missing citations are therefore not a default generation blocker.
    req = make_request()
    gen = ContentResult(body="# Title\n\nBody without citations.", citations=[])
    r = _evaluate(req, gen, api_key=None, model=None)  # origin defaults to instant
    assert r.passed
    assert "tracability" not in r.failed_blocking


def test_excluded_source_blocks_draft():
    # A draft citing a snippet tagged exclude_for_genai fails the Blocking
    # approved-source gate (workbook KPI 03.02.01).
    base = make_request()
    snippets = [s.model_copy(update={"exclude_for_genai": True}) for s in base.snippets]
    req = make_request(snippets=snippets)
    gen = make_generation()
    r = _evaluate(req, gen, api_key=None, model=None)
    assert not r.passed
    assert "approved_source_content_for_genai" in r.failed_blocking


def test_default_gate_allows_minor_blocking_judge_values():
    minor = KPIResult(
        kpi_id="factuality_truthfullness",
        name="Factuality & truthfullness",
        weight="Blocking",
        monitoring="Mandatory",
        value="few",
        tier=2,
        passed=False,
        source="llm",
    )
    severe = minor.model_copy(update={"value": "moderate"})
    assert _gate_blocking([minor], origin="instant") == []
    assert _gate_blocking([severe], origin="instant") == ["factuality_truthfullness"]


def test_tier2_respects_origin_relevance():
    # Groundedness is "Not applicable" for human-authored content in the
    # workbook — that judge must not run for origin=human.
    from aurora_tool_server.evaluation.catalogue import load_catalogue
    from aurora_tool_server.evaluation.tier2_judges import JUDGES, run_tier2

    req, gen = make_request(), make_generation()
    results = run_tier2(
        catalogue=load_catalogue(), req=req, gen=gen,
        api_key=None, model=None, origin="human", channel="web",
    )
    ids = {r.kpi_id for r in results}
    assert "groundedness_source" not in ids
    assert "completeness_source" not in ids
    assert len(results) == len(JUDGES) - 1


def test_no_intent_does_not_fail_the_draft():
    # Raw /v1/evaluations/score callers may not have an intent. The default
    # generation gate no longer runs keyword SEO checks, so this still passes.
    req, gen = make_request(intent=None), make_generation()
    r = _evaluate(req, gen, api_key=None, model=None)
    assert all(k.kpi_id != "h1_header_keywords" for k in r.results)
    assert r.passed
