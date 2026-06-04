"""Per-turn generator: produces clarifying questions / proposed prompt.

LLM path uses ``openai.beta.chat.completions.parse`` with the
:class:`GeneratorOutput` schema. Falls back to a heuristic stub when no
``api_key``/``model`` is configured (so the frontend stays usable on the
deterministic-only path).
"""

from __future__ import annotations

from typing import Sequence

import openai
from loguru import logger

from backend.intent import IntentResult
from backend.prompt_refinement.types import (
    GeneratorOutput,
    QuestionWithChoices,
    RefinementTurn,
)
from backend.retrieval.types import RetrievalResult
from profiles import ProfileBundle


def _summarise_snippets(result: RetrievalResult, limit: int = 5) -> str:
    lines: list[str] = []
    for s in result.snippets[:limit]:
        excerpt = (s.content or "").strip().replace("\n", " ")[:200]
        lines.append(
            f"- [{s.source_doc}::{s.node_id}] {s.title}\n"
            f"    {excerpt}"
        )
    return "\n".join(lines) or "(no snippets retrieved)"


def _summarise_profiles(bundle: ProfileBundle) -> str:
    parts: list[str] = []
    if bundle.workflow:
        parts.append("Workflow profiles:")
        for w in bundle.workflow:
            skills = ", ".join(w.skills[:4]) if w.skills else "—"
            parts.append(f"  - {w.name}: skills={skills}")
    if bundle.domain_expert:
        parts.append("Domain experts:")
        for e in bundle.domain_expert:
            areas = ", ".join(e.expertise_areas[:4]) if e.expertise_areas else "—"
            parts.append(f"  - {e.name} ({e.sector}): {areas}")
    return "\n".join(parts) or "(no profiles selected)"


def _format_prior_turns(turns: Sequence[RefinementTurn], limit: int = 6) -> str:
    if not turns:
        return "(no prior turns)"
    recent = list(turns)[-limit:]
    return "\n".join(
        f"[{t.role}] {t.content}" for t in recent
    )


def _stub_questions(
    intent: IntentResult, result: RetrievalResult
) -> list[QuestionWithChoices]:
    """Cheap heuristic questions with canned choices for the no-LLM fallback."""
    qs: list[QuestionWithChoices] = []
    if not intent.language:
        qs.append(
            QuestionWithChoices(
                question="Which language should the output be in?",
                choices=["English", "Dutch", "Both (bilingual)"],
            )
        )
    if not intent.topic_keywords:
        qs.append(
            QuestionWithChoices(
                question="Which specific angle should we focus on?",
                choices=["Market trends", "Regulatory impact", "Customer impact", "Technical depth"],
            )
        )
    if result.snippets:
        titles = [s.title for s in result.snippets[:3]]
        qs.append(
            QuestionWithChoices(
                question="Any of these candidates you want to prioritise or drop?",
                choices=titles + ["Use all of them"],
            )
        )
    if not qs:
        qs.append(
            QuestionWithChoices(
                question="What audience, length, or tone should we target?",
                choices=[
                    "Board-level execs, ~400 words",
                    "IT directors, ~800 words",
                    "General audience, blog-style",
                ],
            )
        )
    return qs[:3]


_SYSTEM_PROMPT = """You are a refinement assistant for ABN AMRO's editorial co-pilot.

A user submitted a request. The pipeline has already (a) classified intent,
(b) selected workflow + domain-expert profiles, and (c) retrieved candidate
snippets from the corpus. Your job: ask the user 1–3 sharp clarifying
questions to make the prompt sharper, grounded in what the snippets and
profiles actually enable.

Each question MUST come with 2–4 suggested short answers (``choices``) the
user can pick with one click. Choices should be concrete and concise (under
60 chars each). If a question is genuinely open-ended where suggestions
would mislead, return an empty ``choices`` list and the user will type a
free-text reply.

Rules:
- Ask only what would change the output: audience, tone, length, angle,
  inclusion/exclusion of specific retrieved articles, time horizon, region.
- Prefer questions that reference concrete retrieved articles or expert
  areas — vague questions are wasted turns.
- If the refined prompt is already specific enough to generate confidently,
  set ``done=true`` and leave ``questions`` empty.
- Optionally propose a refined prompt (``proposed_prompt``) when the user's
  prior replies make a clear improvement obvious; the user can accept by
  locking in.
- ``reasoning`` is one short sentence for logs — not shown to the user.
"""


def generate_turn(
    *,
    original_prompt: str,
    refined_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundle,
    retrieval: RetrievalResult,
    prior_turns: Sequence[RefinementTurn],
    api_key: str | None = None,
    model: str | None = None,
) -> GeneratorOutput:
    """Produce the next assistant turn — LLM-backed when configured."""
    if not api_key or not model:
        logger.info("Refinement generator: stub path (no api_key/model)")
        return GeneratorOutput(
            questions=_stub_questions(intent, retrieval),
            proposed_prompt=None,
            done=False,
            reasoning="stub generator — no LLM configured",
        )

    user_msg = (
        f"Original prompt:\n  {original_prompt}\n\n"
        f"Current refined prompt:\n  {refined_prompt}\n\n"
        f"Intent: task_codes={intent.task_codes} sector={intent.sector!r} "
        f"topic_keywords={intent.topic_keywords} language={intent.language!r}\n\n"
        f"{_summarise_profiles(profiles)}\n\n"
        f"Retrieved snippets:\n{_summarise_snippets(retrieval)}\n\n"
        f"Prior turns:\n{_format_prior_turns(prior_turns)}\n"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=GeneratorOutput,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no parsed output")
        logger.info(
            f"Refinement generator: LLM path "
            f"(model={model}, questions={len(parsed.questions)}, done={parsed.done})"
        )
        return parsed
    except Exception as e:
        logger.warning(f"Refinement generator: LLM call failed ({e}); using stub")
        return GeneratorOutput(
            questions=_stub_questions(intent, retrieval),
            proposed_prompt=None,
            done=False,
            reasoning=f"stub fallback — LLM error: {e}",
        )
