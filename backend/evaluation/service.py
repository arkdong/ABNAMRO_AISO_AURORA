"""Evaluation stage entry point (Stage 6).

Orchestrates Tier 1 (deterministic) → short-circuit gate → Tier 2 (LLM judges)
and reports the dCLP-step requirements alongside an audit-style maturity
rollup per category.

Two modes, mirroring the other stages:
- LLM mode when both ``api_key`` and ``model`` are supplied: Tier 2 runs.
- Deterministic mode otherwise: Tier 2 emits ``not_evaluated`` records that
  pass by default so the rest of the UI works in dev.

A ``strict_mode=True`` flip turns the default-pass behaviour off — useful
in production to ensure missing-LLM situations don't silently approve
content.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from backend.content_generation.types import ContentRequest, ContentResult
from backend.evaluation.catalogue import Catalogue, load_catalogue
from backend.evaluation.tier1_deterministic import run_tier1
from backend.evaluation.tier2_judges import run_tier2
from backend.evaluation.tier3_human_loop import (
    pending_results,
    required_dclp_steps,
)
from backend.evaluation.types import (
    Channel,
    EvaluationResult,
    KPIResult,
    Origin,
)


def _maturity_for_ratio(ratio: float) -> str:
    if ratio < 0.5:
        return "low"
    if ratio < 0.8:
        return "medium"
    return "high"


def _aggregate_maturity(results: list[KPIResult]) -> dict[str, str]:
    """Per-category rollup using passed/total of leaf KPIs.

    Tier-3 ``pending`` entries are skipped — they aren't a quality verdict,
    just a workflow flag.
    """
    by_cat: dict[str, list[KPIResult]] = {}
    for r in results:
        if r.tier == 3:
            continue
        cat = r.category or "Uncategorised"
        by_cat.setdefault(cat, []).append(r)
    return {
        cat: _maturity_for_ratio(sum(1 for r in rs if r.passed) / len(rs))
        for cat, rs in by_cat.items()
        if rs
    }


def _gate_blocking(results: list[KPIResult]) -> list[str]:
    """IDs of any Mandatory + Blocking KPI that did not pass."""
    return [
        r.kpi_id
        for r in results
        if r.weight == "Blocking" and r.monitoring == "Mandatory" and not r.passed
    ]


def evaluate(
    req: ContentRequest,
    gen: ContentResult,
    *,
    channel: Channel = "web",
    origin: Origin = "instant",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    strict_mode: bool = False,
    catalogue: Optional[Catalogue] = None,
) -> EvaluationResult:
    """Score the generated content against the KPI catalogue.

    Returns an :class:`EvaluationResult`. Never raises — even if Tier 2
    judges all error, the result still carries the Tier 1 verdicts and the
    dCLP-step requirements so the UI can render *something*.
    """
    cat = catalogue or load_catalogue()
    results: list[KPIResult] = []

    # ── Tier 1 — deterministic ─────────────────────────────────────────
    tier1 = run_tier1(
        catalogue=cat, req=req, gen=gen, origin=origin, channel=channel
    )
    results.extend(tier1)

    # Short-circuit: any Blocking failure here means we don't burn LLM
    # spend on Tier 2. Surfacing what's wrong is more important than fully
    # ranking the content.
    early_blockers = _gate_blocking(tier1)
    if early_blockers:
        logger.info(
            f"Evaluation: Tier 1 blocked ({early_blockers}); skipping Tier 2"
        )
        dclp_ids = required_dclp_steps(cat, origin=origin, channel=channel)
        results.extend(pending_results(cat, dclp_ids))
        return EvaluationResult(
            passed=False,
            failed_blocking=early_blockers,
            results=results,
            maturity_by_category=_aggregate_maturity(results),
            dclp_steps_required=dclp_ids,
            channel=channel,
            origin=origin,
            model=None,
            source="deterministic",
            reasoning="Tier 1 short-circuit on blocking KPI failure",
        )

    # ── Tier 2 — LLM judges ────────────────────────────────────────────
    tier2 = run_tier2(
        catalogue=cat, req=req, gen=gen, api_key=api_key, model=model
    )
    # In strict mode, flip ``skipped`` (no-LLM stub) results to failing so
    # absent infra cannot silently approve content.
    if strict_mode:
        for r in tier2:
            if r.source == "skipped":
                r.passed = False
                r.reason = "strict mode: LLM judge required but not configured"
    results.extend(tier2)

    # ── Tier 3 — dCLP requirements ─────────────────────────────────────
    dclp_ids = required_dclp_steps(cat, origin=origin, channel=channel)
    results.extend(pending_results(cat, dclp_ids))

    blockers = _gate_blocking(results)
    overall_source = "llm" if (api_key and model) else "deterministic"
    logger.info(
        f"Evaluation: passed={not blockers} "
        f"(tier1={len(tier1)} kpis, tier2={len(tier2)}, dclp={len(dclp_ids)}, "
        f"blockers={blockers}, source={overall_source})"
    )
    return EvaluationResult(
        passed=not blockers,
        failed_blocking=blockers,
        results=results,
        maturity_by_category=_aggregate_maturity(results),
        dclp_steps_required=dclp_ids,
        channel=channel,
        origin=origin,
        model=model if overall_source == "llm" else None,
        source=overall_source,
        reasoning="evaluated across tier 1+2+3",
    )
