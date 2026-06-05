from __future__ import annotations

from typing import Any

import pytest

from frontend import agent_tools
from frontend.agent_tools import AuroraAgentToolConfig


class FakeAuroraApiClient:
    calls: list[tuple[str, Any]]

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.calls = []

    def close(self) -> None:
        self.calls.append(("close", None))

    def health(self) -> dict[str, str]:
        self.calls.append(("health", self.base_url))
        return {"status": "ok", "service": "aurora-tool-server"}

    def classify_intent(self, user_prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("classify", {"prompt": user_prompt, "options": options}))
        return {
            "role": "Insights Editorial",
            "task_codes": ["T1_DRAFT"],
            "confidence": 0.9,
            "task_reason": "test",
        }

    def select_profiles(
        self,
        intent: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("profiles", {"intent": intent, "options": options}))
        return {"workflow": [], "domain_expert": []}

    def retrieve_context(
        self,
        user_prompt: str,
        intent: dict[str, Any],
        profiles: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "retrieval",
                {
                    "prompt": user_prompt,
                    "intent": intent,
                    "profiles": profiles,
                    "options": options,
                },
            )
        )
        return {
            "query": {"user_prompt": user_prompt, "k": options["k"], "retrieval_backend": options["retrieval_backend"]},
            "snippets": [],
            "provider": options["retrieval_backend"],
            "corpora_searched": [],
        }

    def run_pipeline(
        self,
        user_prompt: str,
        *,
        refinement_policy: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "run",
                {
                    "prompt": user_prompt,
                    "refinement_policy": refinement_policy,
                    "options": options,
                },
            )
        )
        return {"run_id": "run_123", "status": "completed"}

    def get_audit_trace(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("audit", run_id))
        return {"run_id": run_id, "events": []}


def test_agent_tools_call_rest_client(monkeypatch):
    monkeypatch.setattr(agent_tools, "AuroraApiClient", FakeAuroraApiClient)
    config = AuroraAgentToolConfig(
        api_base_url="http://aurora.test",
        retrieval_backend="pageindex",
        k=3,
        channel="web",
        origin="instant",
        strict_mode=True,
    )

    assert agent_tools.aurora_health_check(config)["status"] == "ok"
    assert agent_tools.aurora_classify_intent(config, "Draft it")["task_codes"] == ["T1_DRAFT"]
    profiles = agent_tools.aurora_select_profiles(config, "Draft it")
    retrieval = agent_tools.aurora_retrieve_context(config, "Draft it")
    run = agent_tools.aurora_run_pipeline(config, "Draft it")
    audit = agent_tools.aurora_get_audit_trace(config, "run_123")

    assert profiles["profiles"] == {"workflow": [], "domain_expert": []}
    assert retrieval["retrieval"]["provider"] == "pageindex"
    assert run["run_id"] == "run_123"
    assert audit == {"run_id": "run_123", "events": []}


def test_agent_run_pipeline_validates_refinement_policy(monkeypatch):
    monkeypatch.setattr(agent_tools, "AuroraApiClient", FakeAuroraApiClient)
    config = AuroraAgentToolConfig()

    with pytest.raises(ValueError, match="refinement_policy"):
        agent_tools.aurora_run_pipeline(config, "Draft it", "later")


def test_build_aurora_function_tools_smoke():
    tools = agent_tools.build_aurora_function_tools(AuroraAgentToolConfig())

    assert [tool.name for tool in tools] == [
        "aurora_health_check",
        "aurora_classify_intent",
        "aurora_select_profiles",
        "aurora_retrieve_context",
        "aurora_run_pipeline",
        "aurora_get_audit_trace",
    ]
