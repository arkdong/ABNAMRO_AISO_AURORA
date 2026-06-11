"""Input container for the evaluation stage.

The evaluator consumes three things from upstream stages: the refined prompt
(what the writer was asked to produce), the retrieved snippets (the approved
evidence) and, optionally, the classified intent (for keyword / language
checks). ``EvalRequest`` bundles them so the tier modules keep the
``req.refined_prompt`` / ``req.snippets`` / ``req.intent`` access shape.

``intent`` is optional: raw REST/MCP callers of ``/v1/evaluations/score`` may
evaluate a draft without running intent classification first. Checks that
need intent fields emit ``source="skipped"`` results in that case rather than
failing the draft for missing context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Channel, IntentResult, Snippet


@dataclass(frozen=True)
class EvalRequest:
    refined_prompt: str
    snippets: list[Snippet] = field(default_factory=list)
    intent: IntentResult | None = None
    # Channel selects the workbook's deviant norms (e.g. chat bubbles).
    channel: Channel = "web"
