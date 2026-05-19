"""Data shapes for the content generation stage (Stage 5)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.intent import IntentResult
from backend.retrieval.types import Snippet


class Citation(BaseModel):
    """One source the model relied on, keyed back to a retrieval snippet.

    ``index`` matches the ``[n]`` marker the model emits inline in ``body``.
    ``source_doc`` + ``node_id`` are echoed back so the UI doesn't have to
    re-cross-reference the original snippet list (it can, for highlighting).
    """

    index: int
    source_doc: str
    node_id: str
    title: str


class ContentRequest(BaseModel):
    """Input bundle for :func:`generate_content`.

    ``ProfileBundle`` is a frozen dataclass (not a pydantic model), so this
    request wrapper keeps things loose and lets the service module pull what
    it needs without forcing arbitrary_types_allowed across the package.
    """

    model_config = {"arbitrary_types_allowed": True}

    refined_prompt: str
    intent: IntentResult
    profiles: object  # ProfileBundle — kept untyped to avoid a hard import
    snippets: list[Snippet] = Field(default_factory=list)


class ContentResult(BaseModel):
    """Structured output from the generation LLM call.

    The ``body`` is markdown the UI renders as-is. ``citations`` lists every
    ``[n]`` index referenced in the body. ``source`` mirrors the convention
    used by intent/retrieval (``"llm"`` vs ``"deterministic"``) so callers
    can decide whether to surface the model name.
    """

    body: str
    citations: list[Citation] = Field(default_factory=list)
    model: Optional[str] = None
    source: Literal["llm", "deterministic"] = "llm"
    reasoning: str = ""
