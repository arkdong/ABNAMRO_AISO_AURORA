"""Tier 3 — editorial-process signoff requirements.

Four KPIs in the workbook use the indicator ``completed step yes/no`` and
cannot be evaluated by code at generation time; they record whether the
relevant human stage in the dCLP (digital content lifecycle process) has
been signed off. This module surfaces *which* steps are required so the
frontend / workflow system can display them; it does not attempt to record
their completion (that's outside AURORA).
"""

from __future__ import annotations

from ..schemas import Channel, KPIResult, Origin
from .catalogue import Catalogue
from .indicators import YesNoScale

# These four are the dCLP-step KPIs whose indicator is ``completed step yes/no``.
DCLP_STEP_IDS = (
    "human_expert_check_substance",
    "human_expert_check_compliancy_legal",
    "human_expert_check_content",
    "status_of_evaluation",
)


def required_dclp_steps(
    catalogue: Catalogue, *, origin: Origin, channel: Channel
) -> list[str]:
    """Return the dCLP-step ``kpi_id``s that apply to this content's
    origin/channel. The workflow system is expected to fill these in via a
    separate signoff path."""
    applicable_ids = {k.id for k in catalogue.applicable(origin=origin, channel=channel)}
    return [step for step in DCLP_STEP_IDS if step in applicable_ids]


def pending_results(catalogue: Catalogue, ids: list[str]) -> list[KPIResult]:
    """Emit one ``KPIResult`` per required step in the ``pending`` state, so
    the eval result surfaces what's outstanding even when no signoff has
    been recorded yet."""
    out: list[KPIResult] = []
    for kpi_id in ids:
        try:
            kpi = catalogue.by_id(kpi_id)
        except KeyError:
            continue
        out.append(
            KPIResult(
                kpi_id=kpi.id,
                name=kpi.name,
                cluster=kpi.cluster_short,
                category=kpi.category,
                weight=kpi.weight,  # type: ignore[arg-type]
                monitoring=kpi.monitoring,  # type: ignore[arg-type]
                indicator="YesNoScale",
                value=YesNoScale.no.value,
                reason="dCLP step not yet completed",
                tier=3,
                passed=False,
                source="deterministic",
            )
        )
    return out
