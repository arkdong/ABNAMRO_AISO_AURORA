"""Tier 2 — LLM-as-judge rubrics.

Each judge:
1. Picks the rubric anchor from :mod:`.prompt` and the indicator enum from
   :mod:`.indicators`.
2. Calls the configured LLM via ``client.beta.chat.completions.parse`` with a
   tiny ``JudgeOutput`` schema constrained to the enum.
3. Returns a :class:`KPIResult` whose ``passed`` follows the default
   indicator semantics in :func:`indicators.is_passing`.

Judges are independent — the service runs them in parallel via a thread
pool. Failure of one judge does not block the others; the failing KPI is
marked ``passed=False`` with ``reason="judge error: ..."`` and ``source="llm"``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Callable, NamedTuple, Optional

import openai
from loguru import logger
from pydantic import BaseModel

from backend.content_generation.types import ContentRequest, ContentResult
from backend.evaluation.catalogue import KPI, Catalogue
from backend.evaluation.indicators import (
    INDICATOR_REGISTRY,
    ClarityScale,
    CompletenessScale,
    DeviationScale,
    ErrorScale,
    GroundednessScale,
    PresenceScale,
    RelevanceScale,
    is_passing,
)
from backend.evaluation.prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_message,
)
from backend.evaluation.types import KPIResult


class JudgeSpec(NamedTuple):
    """Bind one rubric name to (a) the KPI it scores, (b) the indicator scale."""

    rubric_name: str
    kpi_id: str
    scale: type[Enum]


# The 12 judges the evaluation layer ships with. The ``kpi_id`` strings
# match slugs in ``backend/evaluation/data/kpi_catalogue.json``.
JUDGES: tuple[JudgeSpec, ...] = (
    # Blocking — these short-circuit on failure when run via the service.
    JudgeSpec("factuality", "factuality_truthfullness", ErrorScale),
    JudgeSpec("truthfullness", "truthfullness", DeviationScale),
    JudgeSpec("relevancy", "relevancy", RelevanceScale),
    JudgeSpec("privacy_security", "privacy_and_security", DeviationScale),
    # High-weight, non-blocking.
    JudgeSpec("groundedness", "groundedness_source", GroundednessScale),
    JudgeSpec("completeness_source", "completeness_source", CompletenessScale),
    JudgeSpec("comprehensiveness", "comprehensiveness_answer", CompletenessScale),
    JudgeSpec("clarity", "clarity", ClarityScale),
    # GenAI search-quality-rater rubrics.
    JudgeSpec("uniqueness_added_value", "body_content_uniqueness", PresenceScale),
    JudgeSpec("demonstrable_expertise", "experience_expertise", PresenceScale),
    JudgeSpec("no_paraphrase", "no_paraphrase", PresenceScale),
    JudgeSpec("no_filler", "no_filler", PresenceScale),
)


def _make_output_model(scale: type[Enum]) -> type[BaseModel]:
    """Build a per-scale ``JudgeOutput`` so ``response_format`` can enforce the enum.

    Pydantic accepts a class-level field with an Enum type and OpenAI's
    structured-output path will translate that to a JSON Schema ``enum`` —
    the model literally cannot pick a value outside the scale.
    """

    class JudgeOutput(BaseModel):
        value: scale  # type: ignore[valid-type]
        reason: str = ""

    JudgeOutput.__name__ = f"JudgeOutput_{scale.__name__}"
    return JudgeOutput


def _allowed_values(scale: type[Enum]) -> list[str]:
    return [m.value for m in scale if m.value != "unknown"]


def _result_from_judge(
    *,
    kpi: KPI,
    rubric_name: str,
    scale: type[Enum],
    value: Enum,
    reason: str,
) -> KPIResult:
    passed = is_passing(scale, value)
    return KPIResult(
        kpi_id=kpi.id,
        name=kpi.name,
        cluster=kpi.cluster_short,
        category=kpi.category,
        weight=kpi.weight,  # type: ignore[arg-type]
        monitoring=kpi.monitoring,  # type: ignore[arg-type]
        indicator=scale.__name__,
        value=value.value,
        reason=reason,
        tier=2,
        passed=passed,
        source="llm",
    )


def _error_result(kpi: KPI, scale: type[Enum], reason: str) -> KPIResult:
    return KPIResult(
        kpi_id=kpi.id,
        name=kpi.name,
        cluster=kpi.cluster_short,
        category=kpi.category,
        weight=kpi.weight,  # type: ignore[arg-type]
        monitoring=kpi.monitoring,  # type: ignore[arg-type]
        indicator=scale.__name__,
        value="unknown",
        reason=reason,
        tier=2,
        passed=False,
        source="llm",
    )


def _skipped_result(kpi: KPI, scale: type[Enum], reason: str) -> KPIResult:
    """Used in stub mode (no API key) so the result envelope still lists the KPI."""
    return KPIResult(
        kpi_id=kpi.id,
        name=kpi.name,
        cluster=kpi.cluster_short,
        category=kpi.category,
        weight=kpi.weight,  # type: ignore[arg-type]
        monitoring=kpi.monitoring,  # type: ignore[arg-type]
        indicator=scale.__name__,
        value="not_evaluated",
        reason=reason,
        tier=2,
        passed=True,  # permissive: don't block dev runs on missing LLM
        source="skipped",
    )


def _run_one_judge(
    *,
    spec: JudgeSpec,
    kpi: KPI,
    req: ContentRequest,
    gen: ContentResult,
    client: openai.OpenAI,
    model: str,
) -> KPIResult:
    output_model = _make_output_model(spec.scale)
    user_msg = build_judge_user_message(
        rubric_name=spec.rubric_name,
        req=req,
        gen=gen,
        allowed_values=_allowed_values(spec.scale),
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=output_model,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("judge returned no parsed output")
    return _result_from_judge(
        kpi=kpi,
        rubric_name=spec.rubric_name,
        scale=spec.scale,
        value=parsed.value,
        reason=parsed.reason or "",
    )


def run_tier2(
    *,
    catalogue: Catalogue,
    req: ContentRequest,
    gen: ContentResult,
    api_key: Optional[str],
    model: Optional[str],
    max_workers: int = 6,
    client_factory: Optional[Callable[[str], openai.OpenAI]] = None,
) -> list[KPIResult]:
    """Run all configured judges, in parallel where possible.

    ``client_factory`` is a hook for tests — production callers leave it
    unset and we instantiate ``openai.OpenAI(api_key=api_key)``.
    """
    # Resolve KPIs once. Specs whose KPI isn't in the catalogue are skipped
    # (they'd correspond to a stale judge → catalogue drift).
    resolved: list[tuple[JudgeSpec, KPI]] = []
    for spec in JUDGES:
        try:
            resolved.append((spec, catalogue.by_id(spec.kpi_id)))
        except KeyError:
            logger.debug(f"Tier 2: judge {spec.rubric_name} has no KPI {spec.kpi_id} in catalogue")

    # Stub path: emit ``skipped`` results so the envelope still lists the KPI.
    if not api_key or not model:
        return [
            _skipped_result(kpi, spec.scale, "no LLM configured (stub mode)")
            for spec, kpi in resolved
        ]

    client = (client_factory or openai.OpenAI)(api_key=api_key)
    results: list[KPIResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_spec = {
            pool.submit(
                _run_one_judge,
                spec=spec,
                kpi=kpi,
                req=req,
                gen=gen,
                client=client,
                model=model,
            ): (spec, kpi)
            for spec, kpi in resolved
        }
        for fut in as_completed(future_to_spec):
            spec, kpi = future_to_spec[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001 — judge failures isolated
                logger.warning(f"Tier 2: judge {spec.rubric_name} failed: {e}")
                results.append(_error_result(kpi, spec.scale, f"judge error: {e}"))

    # Stable order: by rubric registration order, not parallel completion.
    spec_order = {spec.kpi_id: i for i, spec in enumerate(JUDGES)}
    results.sort(key=lambda r: spec_order.get(r.kpi_id, 9999))
    return results
