"""State and pipeline helpers for the Streamlit Normal mode page."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


NORMAL_MESSAGES_KEY = "normal_messages"
NORMAL_PENDING_KEY = "normal_pending"
NORMAL_LATEST_RUN_KEY = "normal_latest_run"
NORMAL_SHOW_DETAILS_KEY = "normal_show_details"


def init_normal_mode_state(state: MutableMapping[str, Any]) -> None:
    state.setdefault(NORMAL_MESSAGES_KEY, [])
    state.setdefault(NORMAL_PENDING_KEY, None)
    state.setdefault(NORMAL_LATEST_RUN_KEY, None)
    state.setdefault(NORMAL_SHOW_DETAILS_KEY, False)


def pipeline_options(state: MutableMapping[str, Any]) -> dict[str, Any]:
    return {
        "k": int(state["retrieval_k"]),
        "retrieval_backend": state["retrieval_backend"],
        "channel": state["channel"],
        "origin": state["origin"],
        "strict_mode": bool(state["strict_mode"]),
    }


def pending_questions(pending: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not pending:
        return []
    refinement = pending.get("refinement") or {}
    return list(refinement.get("questions") or pending.get("questions") or [])


def build_pending_from_run(user_prompt: str, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "user_prompt": user_prompt,
        "intent": run.get("intent"),
        "profiles": run.get("profiles"),
        "retrieval": run.get("retrieval"),
        "refinement": run.get("refinement") or {},
        "audit": run.get("audit"),
        "questions": list((run.get("refinement") or {}).get("questions") or []),
    }


def clarification_message(pending: dict[str, Any]) -> str:
    questions = pending_questions(pending)
    if not questions:
        return "I need one more detail before drafting."
    lines = ["I need a few details before drafting:"]
    lines.extend(f"{idx}. {question.get('question', '').strip()}" for idx, question in enumerate(questions, 1))
    return "\n".join(lines)


def answers_by_question(pending: dict[str, Any], answers: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for question, answer in zip(pending_questions(pending), answers, strict=False):
        text = str(answer).strip()
        if text:
            mapped[str(question.get("question", ""))] = text
    return mapped


def continue_after_clarification(
    client: Any,
    pending: dict[str, Any],
    answers: dict[str, str],
    options: dict[str, Any],
) -> dict[str, Any]:
    run_id = pending.get("run_id")
    refinement = client.refine_prompt(
        pending["user_prompt"],
        intent=pending.get("intent"),
        profiles=pending.get("profiles"),
        retrieval=pending.get("retrieval"),
        answers=answers,
        regenerate_on_pivot=True,
        options=options,
        run_id=run_id,
    )
    active_intent = refinement.get("new_intent") or pending.get("intent")
    active_profiles = refinement.get("profiles") or pending.get("profiles")
    active_retrieval = refinement.get("retrieval") or pending.get("retrieval") or {}
    refined_prompt = refinement.get("refined_prompt") or pending["user_prompt"]
    snippets = active_retrieval.get("snippets", [])

    content = client.generate_draft(
        refined_prompt=refined_prompt,
        intent=active_intent,
        profiles=active_profiles,
        snippets=snippets,
        options=options,
        run_id=run_id,
    )
    evaluation = client.evaluate_draft(
        refined_prompt=refined_prompt,
        content=content,
        intent=active_intent,
        profiles=active_profiles,
        snippets=snippets,
        options=options,
        run_id=run_id,
    )
    audit = client.get_audit_trace(run_id) if run_id else None

    return {
        "run_id": run_id,
        "status": "completed",
        "intent": active_intent,
        "profiles": active_profiles,
        "retrieval": active_retrieval,
        "refinement": refinement,
        "content": content,
        "evaluation": evaluation,
        "audit": audit or pending.get("audit") or {"run_id": run_id, "events": []},
    }


def compact_evaluation_verdict(evaluation: dict[str, Any] | None) -> str:
    if not evaluation:
        return "**Review:** No evaluation was returned."

    failed_blocking = evaluation.get("failed_blocking") or []
    if evaluation.get("passed"):
        line = "**Review:** Passed. No blocking KPI violations detected."
    elif failed_blocking:
        failures = ", ".join(f"`{item}`" for item in failed_blocking)
        line = f"**Review:** Blocked by {len(failed_blocking)} KPI failure(s): {failures}."
    else:
        line = "**Review:** Needs attention. No blocking KPI IDs were returned."

    dclp_steps = evaluation.get("dclp_steps_required") or []
    if dclp_steps:
        line += f"\n\n**Human signoff:** {len(dclp_steps)} dCLP step(s) still required."
    return line


def assistant_message_from_run(run: dict[str, Any]) -> str:
    content = run.get("content") or {}
    body = str(content.get("body") or "").strip()
    if not body:
        body = "_No generated content was returned._"
    return f"{body}\n\n{compact_evaluation_verdict(run.get('evaluation'))}"


def answer_summary(answers: dict[str, str]) -> str:
    if not answers:
        return "Clarification answers submitted."
    lines = ["Clarification answers:"]
    lines.extend(f"- **{question}** {answer}" for question, answer in answers.items())
    return "\n".join(lines)
