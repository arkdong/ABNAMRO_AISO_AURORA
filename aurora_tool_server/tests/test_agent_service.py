from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from frontend import agent_service
from frontend.agent_service import AuroraAgentSettings


class FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.kwargs = kwargs


class FakeLastAgent:
    name = "AURORA Editorial Agent"


class FakeResult:
    final_output = "Drafted with AURORA."
    last_agent = FakeLastAgent()
    run_loop_exception = None

    async def stream_events(self):
        call_item = SimpleNamespace(
            type="tool_call_item",
            raw_item={
                "name": "aurora_retrieve_context",
                "arguments": json.dumps({"user_prompt": "Write an article"}),
                "call_id": "call_1",
            },
            call_id="call_1",
            tool_name="aurora_retrieve_context",
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=call_item,
        )
        output_item = SimpleNamespace(
            type="tool_call_output_item",
            raw_item={"call_id": "call_1"},
            call_id="call_1",
            output={
                "ok": True,
                "result": {
                    "retrieval": {
                        "snippets": [{"node_id": "n1"}],
                        "provider": "pageindex",
                    }
                },
            },
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=output_item,
        )

        refine_call_item = SimpleNamespace(
            type="tool_call_item",
            raw_item={
                "name": "aurora_refine_prompt",
                "arguments": json.dumps({"user_prompt": "Write an article", "answers": {}}),
                "call_id": "call_2",
            },
            call_id="call_2",
            tool_name="aurora_refine_prompt",
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=refine_call_item,
        )
        refine_output_item = SimpleNamespace(
            type="tool_call_output_item",
            raw_item={"call_id": "call_2"},
            call_id="call_2",
            output={
                "ok": True,
                "result": {
                    "questions": [
                        {
                            "question": "Desired length?",
                            "choices": ["~500 words", "~1000 words"],
                        }
                    ]
                },
            },
        )
        yield SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=refine_output_item,
        )

    def to_input_list(self) -> list[dict[str, Any]]:
        return [{"role": "assistant", "content": self.final_output}]


class FakeRunner:
    last_input: Any = None

    @classmethod
    def run_streamed(cls, agent: FakeAgent, run_input: Any, *, max_turns: int) -> FakeResult:
        cls.last_input = {"agent": agent, "run_input": run_input, "max_turns": max_turns}
        return FakeResult()


def test_readiness_error_reports_missing_key(monkeypatch):
    monkeypatch.setattr(agent_service, "AGENTS_IMPORT_ERROR", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert agent_service.readiness_error() == "OPENAI_API_KEY is not set for the Streamlit process."


def test_agent_policy_requires_evaluation_and_single_regeneration():
    instructions = agent_service.AGENT_INSTRUCTIONS

    assert "aurora_classify_intent -> aurora_select_profiles -> aurora_retrieve_context" in instructions
    assert "Pass the exact full output objects from each stage" in instructions
    assert "reconstruct, summarize, or shorten intent" in instructions
    assert "Use aurora_run_pipeline_fallback only" in instructions
    assert "quick full-pipeline run" in instructions
    assert "granular stage fails" in instructions
    assert "For every generated draft, call aurora_evaluate_draft" in instructions
    assert "passed=false or any failed_blocking items" in instructions
    assert "regenerate exactly once" in instructions
    assert "append the" in instructions
    assert "evaluation feedback to the refined prompt" in instructions
    assert "call aurora_evaluate_draft once more" in instructions
    assert "After one regeneration attempt, stop" in instructions
    assert "article_title and source_url as a Markdown link" in instructions
    assert "Do not show corpus names" in instructions
    assert "intent.language as the final draft/output language" in instructions
    assert "Use the user's prompt language for conversational replies" in instructions
    assert "writes in Dutch but asks for an English article" in instructions


def test_extract_clarification_questions_prefers_tool_events():
    events = [
        {
            "tool_name": "aurora_refine_prompt",
            "questions": [
                {"question": "Desired length?", "choices": ["~500 words", "~1000 words"]}
            ],
        }
    ]

    questions = agent_service.extract_clarification_questions("No questions here.", events)

    assert questions == [
        {"question": "Desired length?", "choices": ["~500 words", "~1000 words"]}
    ]


def test_parse_plain_text_clarification_questions():
    content = """I have a few quick clarifying questions before drafting:

What specific aspects should I focus on?
A) Agentic AI capabilities in cybersecurity
B) Vulnerabilities and risks it poses
C) Adoption challenges for TMT companies
D) All of the above
Desired length?
A) ~500 words
B) ~1000 words
C) ~1500 words
Tone?
A) Formal
B) Conversational
C) Mix of both
Please reply with your choices (e.g., 1:D, 2:A, 3:C)."""

    questions = agent_service.parse_clarification_questions(content)

    assert questions == [
        {
            "question": "What specific aspects should I focus on?",
            "choices": [
                "Agentic AI capabilities in cybersecurity",
                "Vulnerabilities and risks it poses",
                "Adoption challenges for TMT companies",
                "All of the above",
            ],
        },
        {
            "question": "Desired length?",
            "choices": ["~500 words", "~1000 words", "~1500 words"],
        },
        {
            "question": "Tone?",
            "choices": ["Formal", "Conversational", "Mix of both"],
        },
    ]


def test_parse_unlabelled_plain_text_clarification_questions():
    content = """I have a few quick clarification questions before drafting:

Which aspects of Agentic AI should I highlight? (pick one or more)
Autonomous offensive strategies
Defensive countermeasures
Impact on existing cybersecurity protocols
Case studies of implementation
Preferred tone?
Formal and technical
Conversational and accessible
Balanced mix of both
Engaging and persuasive
Include workforce-shortage data?
Yes, include recent statistics
Yes, include case studies
No, keep it general
Include qualitative insights without statistics
Reply with your choices (e.g. "1: Autonomous offensive + Defensive; 2: Balanced")."""

    questions = agent_service.parse_clarification_questions(content)

    assert questions == [
        {
            "question": "Which aspects of Agentic AI should I highlight? (pick one or more)",
            "choices": [
                "Autonomous offensive strategies",
                "Defensive countermeasures",
                "Impact on existing cybersecurity protocols",
                "Case studies of implementation",
            ],
            "multiple": True,
        },
        {
            "question": "Preferred tone?",
            "choices": [
                "Formal and technical",
                "Conversational and accessible",
                "Balanced mix of both",
                "Engaging and persuasive",
            ],
        },
        {
            "question": "Include workforce-shortage data?",
            "choices": [
                "Yes, include recent statistics",
                "Yes, include case studies",
                "No, keep it general",
                "Include qualitative insights without statistics",
            ],
        },
    ]


def test_parse_plain_article_questions_are_not_clarification_forms():
    content = """What can companies do?

Segment the network so attackers cannot move freely.
Practise incident response and make sure backups work.
Train employees continuously."""

    assert agent_service.parse_clarification_questions(content) == []


def test_run_agent_turn_uses_runner_and_json_safe_history(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(agent_service, "AGENTS_IMPORT_ERROR", None)
    monkeypatch.setattr(agent_service, "Agent", FakeAgent)
    monkeypatch.setattr(agent_service, "Runner", FakeRunner)
    monkeypatch.setattr(agent_service, "build_aurora_function_tools", lambda config: [])
    settings = AuroraAgentSettings.from_values(
        api_base_url="http://aurora.test",
        model="gpt-test",
        retrieval_backend="pageindex",
        k=3,
        channel="web",
        origin="instant",
        strict_mode=False,
        run_id="run_agent",
    )
    streamed_events = []

    result = agent_service.run_agent_turn(
        "Write an article",
        settings=settings,
        input_items=[{"role": "assistant", "content": "Earlier"}],
        on_tool_event=streamed_events.append,
    )

    assert result.final_output == "Drafted with AURORA."
    assert result.input_items == [{"role": "assistant", "content": "Drafted with AURORA."}]
    assert [event["kind"] for event in result.tool_events] == ["call", "output", "call", "output"]
    assert [event.kind for event in streamed_events] == ["call", "output", "call", "output"]
    assert result.tool_events[0]["tool_name"] == "aurora_retrieve_context"
    assert "k=3" in result.tool_events[0]["summary"]
    assert "snippets=1" in result.tool_events[1]["summary"]
    assert result.tool_events[3]["questions"] == [
        {"question": "Desired length?", "choices": ["~500 words", "~1000 words"]}
    ]
    assert FakeRunner.last_input["max_turns"] == 12
    assert FakeRunner.last_input["agent"].kwargs["model"] == "gpt-test"
    assert FakeRunner.last_input["agent"].kwargs["tools"] == []
    assert FakeRunner.last_input["run_input"][-1] == {
        "role": "user",
        "content": "Write an article",
    }


def test_run_agent_turn_requires_openai_key(monkeypatch):
    monkeypatch.setattr(agent_service, "AGENTS_IMPORT_ERROR", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = AuroraAgentSettings.from_values(
        api_base_url="http://aurora.test",
        retrieval_backend="pageindex",
        k=3,
        channel="web",
        origin="instant",
        strict_mode=False,
    )

    with pytest.raises(agent_service.AuroraAgentConfigurationError):
        agent_service.run_agent_turn("Hello", settings=settings)
