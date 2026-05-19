"""Prompt refinement — backend module (stage 4).

A post-RAG iterative loop that grounds clarifying questions in the snippets
already retrieved plus the profile bundle already selected. The output is a
refined prompt and a flag indicating whether the change is significant enough
to warrant re-running classification, profile selection, and retrieval.

Public surface:

- :class:`RefinementTurn`, :class:`RefinedPrompt`, :class:`GeneratorOutput`,
  :class:`RefinementResult` — data shapes.
- :func:`advance_turn` — produce the next turn given session state.
- :func:`needs_re_retrieval` — does the refined intent diverge from the
  original enough to justify re-running the chain?
"""

from __future__ import annotations

from backend.prompt_refinement.service import (
    advance_turn,
    append_qa,
    needs_re_retrieval,
)
from backend.prompt_refinement.types import (
    GeneratorOutput,
    QuestionWithChoices,
    RefinedPrompt,
    RefinementResult,
    RefinementTurn,
)

__all__ = [
    "GeneratorOutput",
    "QuestionWithChoices",
    "RefinedPrompt",
    "RefinementResult",
    "RefinementTurn",
    "advance_turn",
    "append_qa",
    "needs_re_retrieval",
]
