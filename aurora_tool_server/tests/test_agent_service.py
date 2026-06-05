from __future__ import annotations

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

    def to_input_list(self) -> list[dict[str, Any]]:
        return [{"role": "assistant", "content": self.final_output}]


class FakeRunner:
    last_input: Any = None

    @classmethod
    def run_sync(cls, agent: FakeAgent, run_input: Any, *, max_turns: int) -> FakeResult:
        cls.last_input = {"agent": agent, "run_input": run_input, "max_turns": max_turns}
        return FakeResult()


def test_readiness_error_reports_missing_key(monkeypatch):
    monkeypatch.setattr(agent_service, "AGENTS_IMPORT_ERROR", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert agent_service.readiness_error() == "OPENAI_API_KEY is not set for the Streamlit process."


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
    )

    result = agent_service.run_agent_turn(
        "Write an article",
        settings=settings,
        input_items=[{"role": "assistant", "content": "Earlier"}],
    )

    assert result.final_output == "Drafted with AURORA."
    assert result.input_items == [{"role": "assistant", "content": "Drafted with AURORA."}]
    assert FakeRunner.last_input["max_turns"] == 8
    assert FakeRunner.last_input["agent"].kwargs["model"] == "gpt-test"
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
