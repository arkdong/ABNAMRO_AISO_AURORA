"""Data shapes for the evaluation stage (Stage 6).

Outputs are designed to round-trip into the ABN AMRO content-quality
dashboards — every ``KPIResult`` carries the indicator enum value that
matches the workbook's standardised scale (see :mod:`.indicators`).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Channel = Literal["web", "chat", "messages", "employee", "app_ib"]
"""Content channel — selects channel-specific norms from the catalogue."""

Origin = Literal["human", "genai_knowledge", "instant"]
"""How the content was produced.

- ``human``: hand-authored / scripted content.
- ``genai_knowledge``: generated against a GenAI source (RAG-style; the source
  has been pre-approved / has metadata).
- ``instant``: instantly generated (no curated GenAI source).
"""


class KPIResult(BaseModel):
    """One KPI scored against the generated content."""

    kpi_id: str
    name: str
    cluster: Optional[str]
    category: Optional[str]
    weight: Literal["Blocking", "High", "Medium", "Low"]
    monitoring: Literal["Mandatory", "Optional"]
    indicator: Optional[str] = None  # name of the scale enum, e.g. ``ErrorScale``
    value: str  # enum value, or ``"not_evaluated"`` / ``"unknown"`` sentinels
    raw_metric: Optional[dict] = None  # underlying number/snippet for tier 1
    reason: str = ""
    tier: Literal[1, 2, 3]
    passed: bool
    source: Literal["deterministic", "llm", "skipped"] = "deterministic"


class EvaluationResult(BaseModel):
    """Top-level evaluator output.

    ``passed`` is ``False`` if any Mandatory + Blocking KPI failed.
    ``failed_blocking`` lists the offending ``kpi_id``s for the UI to surface.
    ``maturity_by_category`` is the audit-style rollup (``low/medium/high``)
    used by the existing content dashboards.
    """

    passed: bool
    failed_blocking: list[str] = Field(default_factory=list)
    results: list[KPIResult] = Field(default_factory=list)
    maturity_by_category: dict[str, str] = Field(default_factory=dict)
    dclp_steps_required: list[str] = Field(default_factory=list)
    channel: Channel = "web"
    origin: Origin = "instant"
    model: Optional[str] = None
    source: Literal["deterministic", "llm"] = "deterministic"
    reasoning: str = ""
