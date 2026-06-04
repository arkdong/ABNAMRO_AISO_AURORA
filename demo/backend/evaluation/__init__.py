"""Content evaluation — backend module (Stage 6).

Scores Stage-5 generated content against the ABN AMRO Content KPI inventory
(``data/Content KPI inventory_AISO.xlsx``). Three tiers:

- **Tier 1**: deterministic checks (readability, structure, citation sanity).
- **Tier 2**: LLM-as-judge rubrics (factuality, groundedness, clarity, …).
- **Tier 3**: dCLP human-signoff requirements (declared, not evaluated).

Public surface:

- :class:`EvaluationResult`, :class:`KPIResult` — output shapes.
- :func:`evaluate` — single entry point with stub + LLM paths.
- :func:`load_catalogue` — typed accessor over the generated catalogue JSON.
"""

from __future__ import annotations

from backend.evaluation.catalogue import Catalogue, KPI, load_catalogue
from backend.evaluation.service import evaluate
from backend.evaluation.types import (
    Channel,
    EvaluationResult,
    KPIResult,
    Origin,
)

__all__ = [
    "Catalogue",
    "Channel",
    "EvaluationResult",
    "KPI",
    "KPIResult",
    "Origin",
    "evaluate",
    "load_catalogue",
]
