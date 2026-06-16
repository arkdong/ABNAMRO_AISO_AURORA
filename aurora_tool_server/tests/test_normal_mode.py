from __future__ import annotations

from typing import Any

from frontend.normal_mode import (
    answers_by_question,
    assistant_message_from_run,
    build_pending_from_run,
    clarification_message,
    continue_after_clarification,
    init_normal_mode_state,
)


def _intent() -> dict[str, Any]:
    return {
        "role": "Insights Editorial",
        "task_codes": ["T1_DRAFT"],
        "confidence": 0.9,
        "task_reason": "Draft request",
    }


def _profiles() -> dict[str, Any]:
    return {
        "workflow": [{"id": "drafter", "name": "Drafter"}],
        "domain_expert": [{"id": "expert_secret", "name": "Hidden Expert"}],
    }


def _retrieval() -> dict[str, Any]:
    return {
        "query": {"user_prompt": "Draft it", "k": 3, "retrieval_backend": "pageindex"},
        "provider": "pageindex",
        "corpora_searched": ["corpus_en"],
        "snippets": [
            {
                "source_doc": "source_secret",
                "node_id": "n1",
                "title": "Hidden source",
                "article_title": "Visible linked article",
                "source_url": "https://www.abnamro.nl/visible-linked-article.html",
                "content": "Snippet secret",
                "score": 0.88,
                "reason": "Relevant",
            }
        ],
    }


def _evaluation(*, passed: bool = True) -> dict[str, Any]:
    return {
        "passed": passed,
        "failed_blocking": [] if passed else ["KPI-1"],
        "results": [
            {
                "kpi_id": "KPI-1",
                "name": "Grounding",
                "weight": "Blocking",
                "value": "pass" if passed else "fail",
                "tier": 1,
                "passed": passed,
                "source": "deterministic",
            }
        ],
        "dclp_steps_required": ["Editorial approval"],
    }


def test_init_normal_mode_state_uses_separate_keys():
    state: dict[str, Any] = {}

    init_normal_mode_state(state)

    assert state["normal_messages"] == []
    assert state["normal_pending"] is None
    assert state["normal_latest_run"] is None
    assert state["normal_show_details"] is False


def test_completed_run_renders_final_content_plus_compact_verdict():
    run = {
        "run_id": "run_123",
        "status": "completed",
        "content": {"body": "Draft body [1]", "citations": []},
        "evaluation": _evaluation(passed=True),
    }

    message = assistant_message_from_run(run)

    assert "Draft body [1]" in message
    assert "Review:" in message
    assert "Passed" in message
    assert "Human signoff" in message


def test_completed_run_renders_citation_as_article_title_link():
    run = {
        "run_id": "run_links",
        "status": "completed",
        "retrieval": _retrieval(),
        "content": {
            "body": "Draft body [1]",
            "citations": [
                {
                    "index": 1,
                    "source_doc": "source_secret",
                    "node_id": "n1",
                    "title": "Hidden source",
                    "article_title": "Visible linked article",
                    "source_url": "https://www.abnamro.nl/visible-linked-article.html",
                }
            ],
        },
        "evaluation": _evaluation(passed=True),
    }

    message = assistant_message_from_run(run)

    assert "[Visible linked article](https://www.abnamro.nl/visible-linked-article.html)" in message
    assert "source_secret" not in message
    assert "source_secret::n1" not in message


def test_needs_clarification_run_stores_pending_stage_data_and_questions():
    run = {
        "run_id": "run_clarify",
        "status": "needs_clarification",
        "intent": _intent(),
        "profiles": _profiles(),
        "retrieval": _retrieval(),
        "refinement": {
            "refined_prompt": "Draft it",
            "done": False,
            "questions": [
                {"question": "Who is the audience?", "choices": ["SMEs", "Consumers"]},
                {"question": "Which language?", "choices": ["English", "Dutch"]},
            ],
        },
        "audit": {"run_id": "run_clarify", "events": [{"stage": "intent"}]},
    }

    pending = build_pending_from_run("Draft it", run)
    answers = answers_by_question(pending, ["SMEs", "English"])
    message = clarification_message(pending)

    assert pending["run_id"] == "run_clarify"
    assert pending["intent"] == run["intent"]
    assert pending["profiles"] == run["profiles"]
    assert pending["retrieval"] == run["retrieval"]
    assert len(pending["questions"]) == 2
    assert answers == {"Who is the audience?": "SMEs", "Which language?": "English"}
    assert "Who is the audience?" in message
    assert "Hidden Expert" not in message


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | str | None]] = []

    def refine_prompt(
        self,
        user_prompt: str,
        *,
        intent: dict[str, Any] | None,
        profiles: dict[str, Any] | None,
        retrieval: dict[str, Any] | None,
        answers: dict[str, str],
        regenerate_on_pivot: bool,
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "refine",
                {
                    "user_prompt": user_prompt,
                    "intent": intent,
                    "profiles": profiles,
                    "retrieval": retrieval,
                    "answers": answers,
                    "regenerate_on_pivot": regenerate_on_pivot,
                    "options": options,
                    "run_id": run_id,
                },
            )
        )
        return {
            "refined_prompt": "Refined draft prompt",
            "done": True,
            "questions": [],
            "needs_re_retrieval": False,
        }

    def generate_draft(
        self,
        *,
        refined_prompt: str,
        intent: dict[str, Any],
        profiles: dict[str, Any],
        snippets: list[dict[str, Any]],
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "generate",
                {
                    "refined_prompt": refined_prompt,
                    "intent": intent,
                    "profiles": profiles,
                    "snippets": snippets,
                    "options": options,
                    "run_id": run_id,
                },
            )
        )
        return {"body": "Generated draft [1]", "citations": [{"index": 1}]}

    def evaluate_draft(
        self,
        *,
        refined_prompt: str,
        content: dict[str, Any],
        intent: dict[str, Any],
        profiles: dict[str, Any],
        snippets: list[dict[str, Any]],
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "evaluate",
                {
                    "refined_prompt": refined_prompt,
                    "content": content,
                    "intent": intent,
                    "profiles": profiles,
                    "snippets": snippets,
                    "options": options,
                    "run_id": run_id,
                },
            )
        )
        return _evaluation(passed=True)

    def get_audit_trace(self, run_id: str | None) -> dict[str, Any]:
        self.calls.append(("audit", run_id))
        return {"run_id": run_id, "events": [{"stage": "generation"}]}


def test_answered_clarification_continues_with_pipeline_inspector_payload_shape():
    pending = {
        "run_id": "run_continue",
        "user_prompt": "Draft it",
        "intent": _intent(),
        "profiles": _profiles(),
        "retrieval": _retrieval(),
        "refinement": {
            "questions": [{"question": "Who is the audience?", "choices": ["SMEs"]}]
        },
    }
    options = {"k": 3, "retrieval_backend": "pageindex", "channel": "web"}
    answers = {"Who is the audience?": "SMEs"}
    client = FakeClient()

    run = continue_after_clarification(client, pending, answers, options)

    assert run["status"] == "completed"
    assert run["content"]["body"] == "Generated draft [1]"
    assert [name for name, _payload in client.calls] == [
        "refine",
        "generate",
        "evaluate",
        "audit",
    ]
    refine_payload = client.calls[0][1]
    assert isinstance(refine_payload, dict)
    assert refine_payload["answers"] == answers
    assert refine_payload["regenerate_on_pivot"] is True
    assert refine_payload["run_id"] == "run_continue"

    generate_payload = client.calls[1][1]
    assert isinstance(generate_payload, dict)
    assert generate_payload["refined_prompt"] == "Refined draft prompt"
    assert generate_payload["intent"] == pending["intent"]
    assert generate_payload["profiles"] == pending["profiles"]
    assert generate_payload["snippets"] == pending["retrieval"]["snippets"]
    assert generate_payload["options"] == options


def test_visible_assistant_message_does_not_include_hidden_stage_details():
    run = {
        "run_id": "run_hidden",
        "status": "completed",
        "intent": _intent(),
        "profiles": _profiles(),
        "retrieval": _retrieval(),
        "content": {"body": "Visible draft", "citations": []},
        "evaluation": _evaluation(passed=False),
        "audit": {"run_id": "run_hidden", "events": [{"stage": "retrieval"}]},
    }

    message = assistant_message_from_run(run)

    assert "Visible draft" in message
    assert "KPI-1" in message
    assert run["profiles"]["domain_expert"][0]["id"] == "expert_secret"
    assert run["retrieval"]["snippets"][0]["content"] == "Snippet secret"
    assert "expert_secret" not in message
    assert "Snippet secret" not in message
    assert "source_secret" not in message
