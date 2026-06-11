"""Evaluation stage entry point.

Orchestrates Tier 1 (deterministic) → generation gate → Tier 2 (LLM judges)
→ Tier 3 (dCLP signoff requirements) and reports an audit-style maturity
rollup per category.

Two modes, mirroring the other stages:
- LLM mode when both ``api_key`` and ``model`` are supplied: Tier 2 runs.
- Deterministic mode otherwise: Tier 2 emits ``not_evaluated`` records that
  pass by default so the rest of the UI works in dev.

A ``strict_mode=True`` flip restores fail-closed semantics for missing or
failing Mandatory/Blocking checks. The default mode is intentionally softer:
it blocks only material generation risks from the workbook rather than every
publication, SEO, lifecycle, or style KPI.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..schemas import (
    Channel,
    ContentResult,
    EvaluationResult,
    IntentResult,
    KPIResult,
    Origin,
    Snippet,
)
from .catalogue import Catalogue, load_catalogue
from .inputs import EvalRequest
from .tier1_deterministic import run_tier1
from .tier2_judges import run_tier2
from .tier3_human_loop import pending_results, required_dclp_steps

logger = logging.getLogger(__name__)


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


_DEFAULT_BLOCKING_VALUES: dict[str, set[str]] = {
    # The workbook norm is "no substantial errors"; for generation-time
    # gating, only material errors block. "few" remains visible in the KPI
    # result, but it is treated as an advisory drafting issue.
    "factuality_truthfullness": {"moderate", "several", "numerous"},
    # A draft that is entirely off-topic is not useful to continue with.
    "relevancy": {"off_topic"},
    # The judge prompt reserves "many" for material bank-standard deviations.
    "truthfullness": {"many"},
    "privacy_and_security": {"many"},
    # Explicit source exclusion metadata is a hard, objective stop.
    "approved_source_content_for_genai": {"exclusion"},
}


def _gate_blocking(
    results: list[KPIResult],
    *,
    origin: Origin,
    strict_mode: bool = False,
) -> list[str]:
    """IDs of KPIs that should block the generation stage.

    The workbook contains publication and lifecycle blockers (dCLP signoffs,
    12-month evaluation status, source metadata) alongside content-quality
    blockers. In default mode AURORA blocks only clear generation risks that
    can be judged from the draft itself. Strict mode keeps the old fail-closed
    interpretation for environments that explicitly want it.
    """
    blockers: list[str] = []
    for r in results:
        if r.tier == 3:
            continue
        if (
            strict_mode
            and r.weight == "Blocking"
            and r.monitoring == "Mandatory"
            and not r.passed
        ):
            blockers.append(r.kpi_id)
            continue
        if r.source == "skipped" or r.value == "unknown":
            continue
        if r.kpi_id == "tracability":
            if origin == "genai_knowledge" and r.value == "not_used":
                blockers.append(r.kpi_id)
            continue
        blocking_values = _DEFAULT_BLOCKING_VALUES.get(r.kpi_id)
        if blocking_values and r.value in blocking_values:
            blockers.append(r.kpi_id)
    return blockers


def evaluate_draft(
    *,
    refined_prompt: str,
    content: ContentResult,
    snippets: list[Snippet],
    intent: IntentResult | None = None,
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
    req = EvalRequest(
        refined_prompt=refined_prompt,
        snippets=snippets,
        intent=intent,
        channel=channel,
    )
    gen = content
    results: list[KPIResult] = []

    # ── Tier 1 — deterministic ─────────────────────────────────────────
    tier1 = run_tier1(
        catalogue=cat, req=req, gen=gen, origin=origin, channel=channel
    )
    results.extend(tier1)

    # Short-circuit only on objective generation blockers. Advisory style,
    # SEO, and lifecycle findings continue to Tier 2 so users get useful
    # review signal instead of a brittle red light.
    early_blockers = _gate_blocking(
        tier1, origin=origin, strict_mode=strict_mode
    )
    if early_blockers:
        logger.info(
            "Evaluation: Tier 1 blocked (%s); skipping Tier 2", early_blockers
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
            reasoning="Tier 1 short-circuit on material generation blocker",
        )

    # ── Tier 2 — LLM judges ────────────────────────────────────────────
    tier2 = run_tier2(
        catalogue=cat,
        req=req,
        gen=gen,
        api_key=api_key,
        model=model,
        origin=origin,
        channel=channel,
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

    blockers = _gate_blocking(
        results, origin=origin, strict_mode=strict_mode
    )
    overall_source = "llm" if (api_key and model) else "deterministic"
    logger.info(
        "Evaluation: passed=%s (tier1=%d kpis, tier2=%d, dclp=%d, blockers=%s, source=%s)",
        not blockers, len(tier1), len(tier2), len(dclp_ids), blockers, overall_source,
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
        reasoning="evaluated with softened generation gate across tier 1+2+3",
    )
