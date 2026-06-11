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

``openai`` is imported lazily inside :func:`run_tier2` so the package stays
importable without the dependency and tests can stub ``sys.modules["openai"]``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional

from pydantic import BaseModel

from ..schemas import Channel, ContentResult, KPIResult, Origin
from .catalogue import KPI, Catalogue
from .indicators import (
    CompletenessScale,
    DeviationScale,
    ErrorScale,
    GroundednessScale,
    RelevanceScale,
    is_passing,
)
from .inputs import EvalRequest
from .prompt import JUDGE_SYSTEM_PROMPT, build_judge_user_message

logger = logging.getLogger(__name__)


class JudgeSpec(NamedTuple):
    """Bind one rubric name to (a) the KPI it scores, (b) the indicator scale."""

    rubric_name: str
    kpi_id: str
    scale: type[Enum]


# The default generation-stage judges. The workbook contains many more audit,
# SEO, readability, and lifecycle KPIs; the active LLM set is intentionally
# limited to content-generation essentials.
JUDGES: tuple[JudgeSpec, ...] = (
    # Workbook Blocking content dimensions. The service applies a softened
    # generation gate to these values in default mode.
    JudgeSpec("factuality", "factuality_truthfullness", ErrorScale),
    JudgeSpec("truthfullness", "truthfullness", DeviationScale),
    JudgeSpec("relevancy", "relevancy", RelevanceScale),
    JudgeSpec("privacy_security", "privacy_and_security", DeviationScale),
    # Mandatory/High content-generation dimensions.
    JudgeSpec("groundedness", "groundedness_source", GroundednessScale),
    JudgeSpec("comprehensiveness", "comprehensiveness_answer", CompletenessScale),
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
    req: EvalRequest,
    gen: ContentResult,
    client: Any,
    model: str,
) -> KPIResult:
    output_model = _make_output_model(spec.scale)
    user_msg = build_judge_user_message(
        rubric_name=spec.rubric_name,
        req=req,
        gen=gen,
        allowed_values=_allowed_values(spec.scale),
        kpi=kpi,
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
    req: EvalRequest,
    gen: ContentResult,
    api_key: Optional[str],
    model: Optional[str],
    origin: Origin = "instant",
    channel: Channel = "web",
    max_workers: int = 6,
    client_factory: Optional[Callable[[str], Any]] = None,
) -> list[KPIResult]:
    """Run the configured judges that apply to this origin/channel.

    The workbook's relevance columns exclude some judged KPIs per origin or
    channel (e.g. groundedness/completeness are "Not applicable" for
    human-authored content) — those judges are not run at all.

    ``client_factory`` is a hook for tests — production callers leave it
    unset and we instantiate ``openai.OpenAI(api_key=api_key)``.
    """
    # Resolve KPIs once, honouring the workbook relevance filter. Specs whose
    # KPI isn't in the catalogue are skipped (stale judge → catalogue drift).
    applicable_ids = {
        k.id for k in catalogue.applicable(origin=origin, channel=channel)
    }
    resolved: list[tuple[JudgeSpec, KPI]] = []
    for spec in JUDGES:
        if spec.kpi_id not in applicable_ids:
            logger.debug(
                "Tier 2: judge %s (%s) not applicable for origin=%s channel=%s",
                spec.rubric_name, spec.kpi_id, origin, channel,
            )
            continue
        try:
            resolved.append((spec, catalogue.by_id(spec.kpi_id)))
        except KeyError:
            logger.debug("Tier 2: judge %s has no KPI %s in catalogue", spec.rubric_name, spec.kpi_id)

    # Stub path: emit ``skipped`` results so the envelope still lists the KPI.
    if not api_key or not model:
        return [
            _skipped_result(kpi, spec.scale, "no LLM configured (stub mode)")
            for spec, kpi in resolved
        ]

    if client_factory is None:
        import openai  # resolved per-call so test stubs in sys.modules take effect

        client_factory = openai.OpenAI
    client = client_factory(api_key=api_key)
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
                logger.warning("Tier 2: judge %s failed: %s", spec.rubric_name, e)
                results.append(_error_result(kpi, spec.scale, f"judge error: {e}"))

    # Stable order: by rubric registration order, not parallel completion.
    spec_order = {spec.kpi_id: i for i, spec in enumerate(JUDGES)}
    results.sort(key=lambda r: spec_order.get(r.kpi_id, 9999))
    return results
