"""Data shapes for the prompt refinement stage."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.intent import IntentResult


class QuestionWithChoices(BaseModel):
    """One clarifying question, optionally with 2–4 suggested short answers
    the user can click to pick. Empty ``choices`` means free-text only."""

    question: str
    choices: list[str] = Field(default_factory=list)


class RefinementTurn(BaseModel):
    """One round of the assistant↔user dialog.

    Kept for back-compat with callers that just want a flat history; the
    frontend now drives a structured Q+A log instead (see :class:`QAEntry`
    inside the message state)."""

    role: Literal["assistant", "user"]
    content: str
    proposed_prompt: Optional[str] = None


class RefinedPrompt(BaseModel):
    original: str
    refined: str
    turns_count: int = 0
    locked_in: bool = False


class GeneratorOutput(BaseModel):
    """Structured response from the per-turn generator LLM call."""

    questions: list[QuestionWithChoices] = Field(default_factory=list)
    proposed_prompt: Optional[str] = None
    done: bool = False
    reasoning: str = ""


class RefinementResult(BaseModel):
    """Final state at lock-in. ``new_intent`` is populated after re-classification."""

    prompt: RefinedPrompt
    needs_re_retrieval: bool
    new_intent: Optional[IntentResult] = None
