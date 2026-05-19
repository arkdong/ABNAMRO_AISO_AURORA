"""Refinement orchestration.

- :func:`advance_turn` runs the generator once and packages the result into
  a :class:`RefinementTurn` plus an updated :class:`RefinedPrompt`.
- :func:`needs_re_retrieval` compares the original intent to the intent
  derived from the refined prompt; returns ``True`` when the change is
  substantive enough to justify re-running profile selection + retrieval.
"""

from __future__ import annotations

from typing import Sequence

from backend.intent import IntentResult
from backend.prompt_refinement.generator import generate_turn
from backend.prompt_refinement.types import (
    GeneratorOutput,
    RefinedPrompt,
    RefinementTurn,
)
from backend.retrieval.types import RetrievalResult
from profiles import ProfileBundle

# Jaccard threshold under which topic-keyword sets are considered "different
# enough" to re-retrieve. Empirically 0.5 catches scope pivots without
# triggering on incidental synonyms.
_KEYWORD_JACCARD_PIVOT = 0.5


def advance_turn(
    *,
    original_prompt: str,
    refined_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundle,
    retrieval: RetrievalResult,
    prior_turns: Sequence[RefinementTurn],
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[RefinementTurn, GeneratorOutput]:
    """Produce one assistant turn given the current dialog state."""
    output = generate_turn(
        original_prompt=original_prompt,
        refined_prompt=refined_prompt,
        intent=intent,
        profiles=profiles,
        retrieval=retrieval,
        prior_turns=prior_turns,
        api_key=api_key,
        model=model,
    )
    content_parts: list[str] = []
    if output.questions:
        content_parts.extend(f"- {q}" for q in output.questions)
    if output.proposed_prompt and output.proposed_prompt != refined_prompt:
        content_parts.append(
            f"\nProposed refined prompt:\n> {output.proposed_prompt}"
        )
    if not content_parts:
        content_parts.append("(no further questions; ready to lock in)")
    turn = RefinementTurn(
        role="assistant",
        content="\n".join(content_parts),
        proposed_prompt=output.proposed_prompt,
    )
    return turn, output


def append_qa(
    refined: RefinedPrompt,
    question: str,
    answer: str,
) -> RefinedPrompt:
    """Fold a Q+A pair into the running refined prompt as a clarification.

    Concatenates to a `Clarifications:` block under the original prompt so the
    LLM downstream sees the user's added constraints without losing the
    original intent text. Each entry: ``- <question stem>: <answer>``.
    """
    answer = answer.strip()
    if not answer:
        return refined
    base = (refined.refined or refined.original).rstrip()
    q_stem = question.strip().rstrip("?").strip()
    line = f"- {q_stem}: {answer}"
    if "\nClarifications:\n" in base or base.endswith("Clarifications:"):
        new_text = f"{base}\n{line}"
    else:
        new_text = f"{base}\n\nClarifications:\n{line}"
    return refined.model_copy(
        update={"refined": new_text, "turns_count": refined.turns_count + 1}
    )


def overwrite_prompt(refined: RefinedPrompt, new_text: str) -> RefinedPrompt:
    """Replace the refined prompt verbatim (used by direct-edit path)."""
    return refined.model_copy(
        update={"refined": new_text.strip(), "turns_count": refined.turns_count + 1}
    )


def _kw_set(kws: Sequence[str]) -> set[str]:
    return {k.lower().strip() for k in kws if k and k.strip()}


def needs_re_retrieval(
    original: IntentResult,
    refined: IntentResult,
) -> bool:
    """Return ``True`` when retrieval should be re-run for the refined intent."""
    if set(original.task_codes) != set(refined.task_codes):
        return True
    if (original.sector or "") != (refined.sector or ""):
        return True
    orig_kw = _kw_set(original.topic_keywords)
    new_kw = _kw_set(refined.topic_keywords)
    if orig_kw or new_kw:
        union = orig_kw | new_kw
        if not union:
            return False
        jaccard = len(orig_kw & new_kw) / len(union)
        if jaccard < _KEYWORD_JACCARD_PIVOT:
            return True
    return False
