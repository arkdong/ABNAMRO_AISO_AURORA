"""Three-tier KPI evaluation against ABN AMRO's content KPI catalogue.

Tier 1: deterministic checks (cheap, always run, Blocking failures
short-circuit). Tier 2: schema-constrained LLM judges (optional, parallel).
Tier 3: dCLP human-signoff requirements (declared, never auto-cleared).

Public entry point: :func:`evaluate_draft` — same signature the rest of the
server has always used, plus an optional ``intent``.
"""

from .catalogue import KPI, Catalogue, kpis_by_ids, load_catalogue
from .inputs import EvalRequest
from .service import evaluate_draft
from .tier1_deterministic import CHECK_REGISTRY, run_tier1
from .tier2_judges import JUDGES, run_tier2
from .tier3_human_loop import DCLP_STEP_IDS, pending_results, required_dclp_steps

__all__ = [
    "KPI",
    "Catalogue",
    "kpis_by_ids",
    "load_catalogue",
    "EvalRequest",
    "evaluate_draft",
    "CHECK_REGISTRY",
    "run_tier1",
    "JUDGES",
    "run_tier2",
    "DCLP_STEP_IDS",
    "pending_results",
    "required_dclp_steps",
]
