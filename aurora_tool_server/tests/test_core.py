from __future__ import annotations

import json
import re
import sys
from types import SimpleNamespace

from aurora_tool_server import AuroraConfig, AuroraCore
from aurora_tool_server.schemas import StageOptions


PROMPT = (
    "Write a short analysis article in English about how agentic AI changes "
    "the cybersecurity arms race for Dutch TMT companies."
)


def _core() -> AuroraCore:
    return AuroraCore(
        AuroraConfig(
            intent_api_key=None,
            content_api_key=None,
            evaluation_api_key=None,
            retrieval_k=4,
        )
    )


def test_deterministic_pipeline_completes_with_audit():
    core = _core()
    result = core.run_pipeline(PROMPT)

    assert result.status == "completed"
    assert result.run_id.startswith("run_")
    assert result.intent.task_codes[0] == "T1_DRAFT"
    assert result.intent.sector == "Technologie, Media & Telecom"
    assert result.profiles.workflow
    assert result.retrieval.snippets
    assert result.content is not None
    assert result.content.source == "deterministic"
    assert result.content.citations
    assert result.evaluation is not None
    assert result.evaluation.results

    stages = [event.stage for event in result.audit.events]
    assert stages[:4] == ["intent", "profiles", "retrieval", "refinement"]
    assert "generation" in stages
    assert "evaluation" in stages
    assert stages[-1] == "run_pipeline"


def test_profile_selection_routes_to_cyber_expert():
    core = _core()
    intent = core.classify_intent(PROMPT)
    profiles = core.select_profiles(intent)

    workflow_ids = {profile.id for profile in profiles.workflow}
    expert_ids = {profile.id for profile in profiles.domain_expert}
    assert "drafter" in workflow_ids
    assert "expert_tmt_cybersecurity" in expert_ids


def test_retrieval_child_snippets_carry_article_link_metadata():
    core = _core()
    prompt = "Write about IT sector growth by 4.5 per cent in 2020 and 5G."
    intent = core.classify_intent(prompt)
    profiles = core.select_profiles(intent)

    retrieval = core.retrieve_context(prompt, intent, profiles, options=StageOptions(k=5))

    child = next(
        snippet
        for snippet in retrieval.snippets
        if snippet.title != snippet.article_title and snippet.source_url
    )
    assert child.title == "IT sector to grow by 4.5 per cent in 2020"
    assert child.article_title == "5G essential for the Netherlands’ earning capacity"
    assert child.source_url.endswith("5g-essentieel-voor-het-verdienvermogen-van-nederland.html")


def test_vector_retrieval_recovers_article_link_metadata_for_child_chunks():
    core = _core()
    prompt = "Write about IT sector growth by 4.5 per cent in 2020 and 5G."
    intent = core.classify_intent(prompt)
    profiles = core.select_profiles(intent)

    retrieval = core.retrieve_context(
        prompt,
        intent,
        profiles,
        options=StageOptions(k=5, retrieval_backend="vector_rag"),
    )

    child = next(
        snippet
        for snippet in retrieval.snippets
        if snippet.title != snippet.article_title and snippet.source_url
    )
    assert child.article_title
    assert child.source_url.startswith("https://www.abnamro.nl/")


def test_ask_first_pipeline_stops_for_clarification():
    core = _core()
    result = core.run_pipeline("Draft something about technology.", refinement_policy="ask_first")

    assert result.status == "needs_clarification"
    assert result.content is None
    assert result.evaluation is None
    assert result.refinement.done is False
    assert result.refinement.questions


def test_audit_trace_can_be_read_after_run():
    core = _core()
    result = core.run_pipeline(PROMPT)
    trace = core.get_audit_trace(result.run_id)

    assert trace.run_id == result.run_id
    assert len(trace.events) == len(result.audit.events)


def test_pipeline_can_append_to_external_run_id():
    core = _core()
    run_id = "run_external"
    result = core.run_pipeline(PROMPT, run_id=run_id)

    assert result.run_id == run_id
    assert {event.run_id for event in result.audit.events} == {run_id}


class _FakeOpenAICompletions:
    def create(self, *, messages, **_kwargs):
        system = messages[0]["content"]
        if "Classify this ABN AMRO editorial request" in system:
            content = {
                "role": "Insights Editorial",
                "task_codes": ["T1_DRAFT"],
                "confidence": 0.94,
                "task_reason": "The request asks for a grounded editorial article.",
                "sector": "Technologie, Media & Telecom",
                "topic_keywords": ["agentic ai", "cybersecurity"],
                "language": "en",
            }
        elif "select AURORA workflow" in system:
            content = {
                "workflow_ids": ["drafter"],
                "domain_expert_ids": ["expert_tmt_cybersecurity"],
                "reasons": {
                    "drafter": "The user asked for new editorial content.",
                    "expert_tmt_cybersecurity": "The topic is cybersecurity in TMT.",
                },
                "reasoning": "Selected drafting and cybersecurity expertise.",
            }
        else:
            body = (
                "Agentic AI is reshaping the cybersecurity arms race for TMT firms. "
                "Attackers can automate reconnaissance and exploit chains, while defenders "
                "can use agents to monitor weak signals and coordinate response [1]. "
                "The safest message is resilience: prepare controls, rehearse incidents, "
                "and keep humans in the loop."
            )
            content = {
                "body": body,
                "reasoning": "Generated from the retrieved AURORA evidence.",
                "citation_indices": [1],
            }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )

    def parse(self, *, messages, response_format, **_kwargs):
        name = response_format.__name__
        user_message = messages[-1]["content"]
        if name == "_LLMRanking":
            ids = re.findall(r"\[([^\]]+)\]", user_message)
            parsed = response_format(
                picks=[
                    {
                        "node_id": ids[0],
                        "score": 0.91,
                        "reason": "The section directly supports the requested angle.",
                    }
                ]
            )
        elif name == "_LLMRefinementOutput":
            parsed = response_format(
                questions=[],
                proposed_prompt=(
                    "Write an English analysis article about agentic AI and "
                    "cybersecurity for Dutch TMT companies, grounded in retrieved sources."
                ),
                done=True,
                reasoning="The prompt was already specific enough to lock.",
            )
        elif name.startswith("JudgeOutput_"):
            from aurora_tool_server.evaluation.indicators import PASSING_VALUES

            scale_cls = response_format.model_fields["value"].annotation
            passing = PASSING_VALUES.get(scale_cls)
            value = next(iter(passing)) if passing else list(scale_cls)[0]
            parsed = response_format(value=value, reason="Judge passes in test.")
        else:
            raise AssertionError(f"Unexpected response format {name}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


class _FakeOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        completions = _FakeOpenAICompletions()
        self.chat = SimpleNamespace(completions=completions)
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_llm_config_routes_all_pipeline_stages(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    core = AuroraCore(
        AuroraConfig(
            intent_api_key="sk-test",
            intent_model="gpt-test",
            profile_api_key="sk-test",
            profile_model="gpt-test",
            retrieval_api_key="sk-test",
            retrieval_model="gpt-test",
            refinement_api_key="sk-test",
            refinement_model="gpt-test",
            content_api_key="sk-test",
            content_model="gpt-test",
            evaluation_api_key="sk-test",
            evaluation_model="gpt-test",
            retrieval_k=1,
        )
    )

    result = core.run_pipeline(PROMPT)

    assert result.intent.source == "llm"
    assert result.profiles.source == "llm"
    assert result.retrieval.source == "llm"
    assert result.refinement.source == "llm"
    assert result.content is not None
    assert result.content.source == "llm"
    assert result.evaluation is not None
    assert result.evaluation.source == "llm"

    from aurora_tool_server.evaluation.tier2_judges import JUDGES

    tier2 = [r for r in result.evaluation.results if r.tier == 2]
    assert len(tier2) == len(JUDGES)
    assert all(r.source == "llm" for r in tier2)
    assert all(r.passed for r in tier2)
