from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from aurora_tool_server.generation import generate_draft
from aurora_tool_server.schemas import ProfileBundleResult
from tests.eval_fixtures import make_intent, make_snippets


class _BodyMarkdownCompletions:
    def create(self, **_kwargs):
        content = {
            "body_markdown": "LLM draft body [1].",
            "reasoning": "generated with body_markdown",
            "citation_indices": [1],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )


class _BodyMarkdownOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_BodyMarkdownCompletions())


class _BlankBodyCompletions:
    def create(self, **_kwargs):
        content = {
            "body": "   ",
            "reasoning": "empty response",
            "citation_indices": [],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )


class _BlankBodyOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_BlankBodyCompletions())


def test_llm_body_markdown_key_returns_llm_result(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_BodyMarkdownOpenAI))

    result = generate_draft(
        refined_prompt="Draft a short article about agentic AI.",
        intent=make_intent(),
        profiles=ProfileBundleResult(),
        snippets=make_snippets(),
        api_key="sk-test",
        model="gpt-test",
    )

    assert result.source == "llm"
    assert result.model == "gpt-test"
    assert result.body == "LLM draft body [1]."
    assert len(result.citations) == 1
    assert result.citations[0].node_id == "0114"


def test_blank_llm_body_falls_back_to_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_BlankBodyOpenAI))

    result = generate_draft(
        refined_prompt="Draft a short article about agentic AI.",
        intent=make_intent(),
        profiles=ProfileBundleResult(),
        snippets=make_snippets(),
        api_key="sk-test",
        model="gpt-test",
    )

    assert result.source == "deterministic"
    assert result.body.strip()
    assert "AURORA grounded draft" in result.body
    assert "empty body" in result.reasoning
