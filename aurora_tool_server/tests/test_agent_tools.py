from __future__ import annotations

from typing import Any

import pytest

from frontend import agent_tools
from frontend.agent_tools import AuroraAgentToolConfig


FULL_PROFILES = {
    "workflow": [
        {
            "id": "drafter",
            "name": "Drafter",
            "description": "Writes grounded editorial drafts.",
            "category": "workflow",
            "selection_reason": "Matches draft requests.",
        }
    ],
    "domain_expert": [],
    "source": "deterministic",
    "reasoning": "test",
}


@pytest.fixture(autouse=True)
def clear_stage_cache():
    agent_tools.clear_agent_tool_cache()
    yield
    agent_tools.clear_agent_tool_cache()


class FakeAuroraApiClient:
    calls: list[tuple[str, Any]]
    all_calls: list[tuple[str, Any]] = []

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.calls = []

    def close(self) -> None:
        self._record("close", None)

    def _record(self, name: str, payload: Any) -> None:
        self.calls.append((name, payload))
        self.all_calls.append((name, payload))

    def health(self) -> dict[str, str]:
        self._record("health", self.base_url)
        return {"status": "ok", "service": "aurora-tool-server"}

    def classify_intent(
        self,
        user_prompt: str,
        options: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record("classify", {"prompt": user_prompt, "options": options, "run_id": run_id})
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
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record("profiles", {"intent": intent, "options": options, "run_id": run_id})
        return {"workflow": [], "domain_expert": []}

    def retrieve_context(
        self,
        user_prompt: str,
        intent: dict[str, Any],
        profiles: dict[str, Any],
        options: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record(
            "retrieval",
            (
                {
                "prompt": user_prompt,
                "intent": intent,
                "profiles": profiles,
                "options": options,
                "run_id": run_id,
            }
            ),
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
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record(
            "run",
            {
                "prompt": user_prompt,
                "refinement_policy": refinement_policy,
                "options": options,
                "run_id": run_id,
            },
        )
        return {"run_id": "run_123", "status": "completed"}

    def refine_prompt(
        self,
        user_prompt: str,
        *,
        intent: dict[str, Any] | None,
        profiles: dict[str, Any] | None,
        retrieval: dict[str, Any] | None,
        answers: dict[str, str] | None,
        regenerate_on_pivot: bool,
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record(
            "refine",
            {
                "prompt": user_prompt,
                "intent": intent,
                "profiles": profiles,
                "retrieval": retrieval,
                "answers": answers,
                "regenerate_on_pivot": regenerate_on_pivot,
                "options": options,
                "run_id": run_id,
            },
        )
        return {
            "original_prompt": user_prompt,
            "refined_prompt": user_prompt,
            "questions": [],
            "done": True,
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
        self._record(
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
        return {"body": "Draft body", "citations": []}

    def evaluate_draft(
        self,
        *,
        refined_prompt: str,
        content: dict[str, Any],
        intent: dict[str, Any] | None,
        profiles: dict[str, Any] | None,
        snippets: list[dict[str, Any]],
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record(
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
        return {"passed": True, "failed_blocking": [], "results": []}

    def get_audit_trace(self, run_id: str) -> dict[str, Any]:
        self._record("audit", run_id)
        return {"run_id": run_id, "events": []}


class FullProfileFakeAuroraApiClient(FakeAuroraApiClient):
    all_calls: list[tuple[str, Any]] = []

    def select_profiles(
        self,
        intent: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        self._record("profiles", {"intent": intent, "options": options, "run_id": run_id})
        return FULL_PROFILES


def test_agent_tools_call_rest_client(monkeypatch):
    monkeypatch.setattr(agent_tools, "AuroraApiClient", FakeAuroraApiClient)
    FakeAuroraApiClient.all_calls = []
    config = AuroraAgentToolConfig(
        api_base_url="http://aurora.test",
        retrieval_backend="pageindex",
        k=3,
        channel="web",
        origin="instant",
        strict_mode=True,
        run_id="run_agent",
    )

    assert agent_tools.aurora_health_check(config)["status"] == "ok"
    intent = agent_tools.aurora_classify_intent(config, "Draft it")
    assert intent["task_codes"] == ["T1_DRAFT"]
    profiles = agent_tools.aurora_select_profiles(config, intent)
    retrieval = agent_tools.aurora_retrieve_context(config, "Draft it", intent, profiles)
    run = agent_tools.aurora_run_pipeline(config, "Draft it")
    refinement = agent_tools.aurora_refine_prompt(
        config,
        "Draft it",
        intent=intent,
        profiles=profiles,
        retrieval=retrieval,
        answers={"Audience?": "CFOs"},
    )
    content = agent_tools.aurora_generate_draft(
        config,
        "Draft it",
        intent,
        profiles,
        retrieval["snippets"],
    )
    evaluation = agent_tools.aurora_evaluate_draft(
        config,
        "Draft it",
        content,
        intent,
        profiles,
        retrieval["snippets"],
    )
    audit = agent_tools.aurora_get_audit_trace(config)

    assert profiles == {"workflow": [], "domain_expert": []}
    assert retrieval["provider"] == "pageindex"
    assert run["run_id"] == "run_123"
    assert refinement["done"] is True
    assert content["body"] == "Draft body"
    assert evaluation["passed"] is True
    assert audit == {"run_id": "run_agent", "events": []}
    stage_calls = [
        payload
        for name, payload in FakeAuroraApiClient.all_calls
        if name not in {"health", "audit", "close"}
    ]
    assert stage_calls
    assert all(payload["run_id"] == "run_agent" for payload in stage_calls)
    assert [name for name, _payload in FakeAuroraApiClient.all_calls if name != "close"] == [
        "health",
        "classify",
        "profiles",
        "retrieval",
        "run",
        "refine",
        "generate",
        "evaluate",
        "audit",
    ]


def test_agent_tools_repair_partial_profile_handoffs_from_run_cache(monkeypatch):
    monkeypatch.setattr(agent_tools, "AuroraApiClient", FullProfileFakeAuroraApiClient)
    FullProfileFakeAuroraApiClient.all_calls = []
    config = AuroraAgentToolConfig(api_base_url="http://aurora.test", run_id="run_repair")
    intent = {
        "role": "Insights Editorial",
        "task_codes": ["T1_DRAFT"],
        "confidence": 0.9,
        "task_reason": "test",
    }

    agent_tools.aurora_select_profiles(config, intent)
    partial_profiles = {
        "workflow": [{"id": "drafter", "name": "Drafter"}],
        "domain_expert": [],
    }
    refinement = agent_tools.aurora_refine_prompt(
        config,
        "Draft it",
        intent=intent,
        profiles=partial_profiles,
        answers={},
    )

    refine_payload = next(
        payload
        for name, payload in FullProfileFakeAuroraApiClient.all_calls
        if name == "refine"
    )
    assert refinement["done"] is True
    assert refine_payload["profiles"] == FULL_PROFILES


def test_agent_tool_cache_can_be_cleared_for_one_run(monkeypatch):
    monkeypatch.setattr(agent_tools, "AuroraApiClient", FullProfileFakeAuroraApiClient)
    FullProfileFakeAuroraApiClient.all_calls = []
    config = AuroraAgentToolConfig(api_base_url="http://aurora.test", run_id="run_clear")
    intent = {
        "role": "Insights Editorial",
        "task_codes": ["T1_DRAFT"],
        "confidence": 0.9,
        "task_reason": "test",
    }
    partial_profiles = {
        "workflow": [{"id": "drafter", "name": "Drafter"}],
        "domain_expert": [],
    }

    agent_tools.aurora_select_profiles(config, intent)
    agent_tools.clear_agent_tool_cache("run_clear")
    agent_tools.aurora_refine_prompt(
        config,
        "Draft it",
        intent=intent,
        profiles=partial_profiles,
        answers={},
    )

    refine_payload = next(
        payload
        for name, payload in FullProfileFakeAuroraApiClient.all_calls
        if name == "refine"
    )
    assert refine_payload["profiles"] is None


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
        "aurora_run_pipeline_fallback",
        "aurora_refine_prompt",
        "aurora_generate_draft",
        "aurora_evaluate_draft",
        "aurora_get_audit_trace",
    ]
