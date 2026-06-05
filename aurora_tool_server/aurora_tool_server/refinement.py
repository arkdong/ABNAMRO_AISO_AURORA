"""Prompt refinement helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from .intent import classify_intent
from .profiles import select_profiles
from .retrieval import build_query, retrieve_context
from .schemas import (
    IntentResult,
    ProfileBundleResult,
    RefinementQuestion,
    RefinementResult,
    RetrievalResult,
)

_KEYWORD_JACCARD_PIVOT = 0.5


class _LLMQuestion(BaseModel):
    question: str
    choices: list[str] = Field(default_factory=list)


class _LLMRefinementOutput(BaseModel):
    questions: list[_LLMQuestion] = Field(default_factory=list)
    proposed_prompt: str | None = None
    done: bool = False
    reasoning: str = ""


_SYSTEM_PROMPT = """You are a refinement assistant for ABN AMRO's AURORA editorial co-pilot.

The pipeline already classified intent, selected profiles, and retrieved
candidate snippets. Ask 1-3 sharp clarifying questions that would materially
improve the final output, or set done=true when the prompt is already specific.

Each question should include 2-4 concise choices when useful. Ground questions
in the retrieved snippets, profile expertise, audience, tone, length, angle,
time horizon, language, or source prioritization. Avoid vague questions.

Return structured output with:
- questions: list of question/choices objects
- proposed_prompt: optional improved prompt
- done: true only when no further clarification is needed
- reasoning: one short sentence for audit logs
"""

_LOCK_PROMPT = """You refine AURORA editorial prompts.

Given the original prompt, the clarification answers, intent, profiles, and
retrieved context, produce one concise final prompt for content generation.
Preserve the user's intent, add only constraints supported by the answers, and
do not invent requirements. Return done=true and proposed_prompt set.
"""


def build_questions(intent: IntentResult, retrieval: RetrievalResult | None) -> list[RefinementQuestion]:
    questions: list[RefinementQuestion] = []
    if intent.language is None:
        questions.append(
            RefinementQuestion(
                question="Which output language should AURORA use?",
                choices=["English", "Dutch", "Both English and Dutch"],
            )
        )
    if not any("audience" in kw.lower() for kw in intent.topic_keywords):
        questions.append(
            RefinementQuestion(
                question="Who is the target audience for the content?",
                choices=["Business decision-makers", "IT directors", "General clients"],
            )
        )
    if retrieval is not None and not retrieval.snippets:
        questions.append(
            RefinementQuestion(
                question="No strong source snippets were found. Should AURORA proceed anyway?",
                choices=["Proceed with writing-guide context", "Narrow the topic", "Stop for review"],
            )
        )
    if not questions:
        questions.append(
            RefinementQuestion(
                question="Should AURORA keep the current scope and retrieved sources?",
                choices=["Keep current scope", "Narrow the angle", "Broaden the angle"],
            )
        )
    return questions[:3]


def _summarise_profiles(profiles: ProfileBundleResult) -> str:
    parts: list[str] = []
    if profiles.workflow:
        parts.append("Workflow profiles:")
        for profile in profiles.workflow:
            skills = ", ".join(profile.skills[:5]) or "-"
            parts.append(f"- {profile.name} ({profile.id}): skills={skills}")
    if profiles.domain_expert:
        parts.append("Domain experts:")
        for profile in profiles.domain_expert:
            areas = ", ".join(profile.expertise_areas[:5]) or "-"
            parts.append(f"- {profile.name} ({profile.id}): areas={areas}")
    return "\n".join(parts) or "(no profiles selected)"


def _summarise_retrieval(retrieval: RetrievalResult, limit: int = 5) -> str:
    lines: list[str] = []
    for snippet in retrieval.snippets[:limit]:
        excerpt = (snippet.content or "").strip().replace("\n", " ")[:240]
        lines.append(
            f"- [{snippet.source_doc}::{snippet.node_id}] {snippet.title}\n"
            f"  reason={snippet.reason}\n"
            f"  excerpt={excerpt}"
        )
    return "\n".join(lines) or "(no snippets retrieved)"


def _llm_refinement_turn(
    *,
    system_prompt: str,
    user_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundleResult,
    retrieval: RetrievalResult,
    answers: Mapping[str, str] | None,
    api_key: str,
    model: str,
) -> _LLMRefinementOutput:
    from openai import OpenAI

    answers_block = "\n".join(
        f"- {question}: {answer}" for question, answer in (answers or {}).items()
    ) or "(no answers yet)"
    user_message = (
        f"Original prompt:\n{user_prompt}\n\n"
        f"Answers so far:\n{answers_block}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"Profiles:\n{_summarise_profiles(profiles)}\n\n"
        f"Retrieved snippets:\n{_summarise_retrieval(retrieval)}\n"
    )
    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=_LLMRefinementOutput,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM returned no parsed refinement output")
    return parsed


def _questions_from_llm(output: _LLMRefinementOutput) -> list[RefinementQuestion]:
    return [
        RefinementQuestion(
            question=item.question,
            choices=[choice for choice in item.choices if choice.strip()][:4],
        )
        for item in output.questions[:3]
        if item.question.strip()
    ]


def apply_answers(user_prompt: str, answers: Mapping[str, str]) -> str:
    cleaned = [(q.strip(), a.strip()) for q, a in answers.items() if a and a.strip()]
    if not cleaned:
        return user_prompt.strip()
    lines = [user_prompt.strip(), "", "Clarifications:"]
    lines.extend(f"- {question.rstrip('?')}: {answer}" for question, answer in cleaned)
    return "\n".join(lines)


def _keyword_set(intent: IntentResult) -> set[str]:
    return {kw.lower().strip() for kw in intent.topic_keywords if kw.strip()}


def needs_re_retrieval(original: IntentResult, refined: IntentResult) -> bool:
    if set(original.task_codes) != set(refined.task_codes):
        return True
    if (original.sector or "") != (refined.sector or ""):
        return True
    original_keywords = _keyword_set(original)
    refined_keywords = _keyword_set(refined)
    if original_keywords or refined_keywords:
        union = original_keywords | refined_keywords
        if union and len(original_keywords & refined_keywords) / len(union) < _KEYWORD_JACCARD_PIVOT:
            return True
    return False


def refine_prompt(
    *,
    user_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundleResult,
    retrieval: RetrievalResult,
    answers: Mapping[str, str] | None = None,
    regenerate_on_pivot: bool = False,
    ask_questions: bool = True,
    api_key: str | None = None,
    model: str | None = None,
    profile_api_key: str | None = None,
    profile_model: str | None = None,
    retrieval_api_key: str | None = None,
    retrieval_model: str | None = None,
    k: int = 5,
    retrieval_backend: str = "pageindex",
) -> RefinementResult:
    answers = answers or {}
    if ask_questions and not answers:
        if api_key and model:
            try:
                output = _llm_refinement_turn(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    intent=intent,
                    profiles=profiles,
                    retrieval=retrieval,
                    answers=None,
                    api_key=api_key,
                    model=model,
                )
                return RefinementResult(
                    original_prompt=user_prompt,
                    refined_prompt=output.proposed_prompt or user_prompt.strip(),
                    questions=_questions_from_llm(output),
                    done=output.done and not output.questions,
                    source="llm",
                    reasoning=output.reasoning or "LLM generated refinement questions.",
                )
            except Exception as exc:
                return RefinementResult(
                    original_prompt=user_prompt,
                    refined_prompt=user_prompt.strip(),
                    questions=build_questions(intent, retrieval),
                    done=False,
                    source="deterministic",
                    reasoning=f"LLM refinement failed; deterministic fallback: {exc}",
                )
        return RefinementResult(
            original_prompt=user_prompt,
            refined_prompt=user_prompt.strip(),
            questions=build_questions(intent, retrieval),
            done=False,
            source="deterministic",
            reasoning="Clarification questions generated before lock-in.",
        )

    refined_text = apply_answers(user_prompt, answers)
    source = "deterministic"
    reasoning = "Prompt locked after deterministic refinement."
    if api_key and model:
        try:
            output = _llm_refinement_turn(
                system_prompt=_LOCK_PROMPT,
                user_prompt=user_prompt,
                intent=intent,
                profiles=profiles,
                retrieval=retrieval,
                answers=answers,
                api_key=api_key,
                model=model,
            )
            if output.proposed_prompt:
                refined_text = output.proposed_prompt.strip()
            source = "llm"
            reasoning = output.reasoning or "LLM locked refined prompt from clarifications."
        except Exception as exc:
            reasoning = f"LLM lock failed; deterministic refinement fallback: {exc}"

    new_intent = classify_intent(refined_text, api_key=api_key, model=model)
    pivot = needs_re_retrieval(intent, new_intent)
    new_profiles = None
    new_retrieval = None
    if pivot and regenerate_on_pivot:
        new_profiles = select_profiles(
            new_intent,
            api_key=profile_api_key or api_key,
            model=profile_model or model,
        )
        query = build_query(
            refined_text,
            new_intent,
            new_profiles,
            k=k,
            retrieval_backend=retrieval_backend,
        )
        new_retrieval = retrieve_context(
            query,
            api_key=retrieval_api_key or api_key,
            model=retrieval_model or model,
        )

    return RefinementResult(
        original_prompt=user_prompt,
        refined_prompt=refined_text,
        questions=[],
        done=True,
        needs_re_retrieval=pivot,
        new_intent=new_intent,
        retrieval=new_retrieval,
        profiles=new_profiles,
        source=source,  # type: ignore[arg-type]
        reasoning=reasoning,
    )
