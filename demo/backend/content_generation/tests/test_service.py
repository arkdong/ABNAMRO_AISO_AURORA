"""Smoke tests for the content generation stage.

Covers the three branches of :func:`generate_content`:
1. Stub path (no api_key/model) returns a deterministic placeholder.
2. LLM path with a mocked client returns the parsed result + validates
   citations against the snippet list.
3. Error path falls back to the stub instead of raising.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.content_generation import (
    Citation,
    ContentRequest,
    ContentResult,
    generate_content,
)
from backend.intent import IntentResult
from backend.retrieval.types import Snippet
from profiles import ProfileBundle


def _req(**overrides) -> ContentRequest:
    snippets = [
        Snippet(
            source_doc="corpus_en",
            node_id="0114",
            title="The two faces of Agentic AI",
            content="Agentic AI is both attacker and defender.",
            score=1.0,
            reason="title match",
        ),
        Snippet(
            source_doc="corpus_en",
            node_id="0079",
            title="Cybersecurity in TMT",
            content="The TMT sector's cyber posture is shifting.",
            score=0.9,
            reason="title match",
        ),
    ]
    defaults = dict(
        refined_prompt="Write a 400-word briefing on agentic AI for IT directors.",
        intent=IntentResult(
            role="Insights Editorial",
            task_codes=["T1_DRAFT"],
            confidence=0.9,
            task_reason="testing",
            sector="Technologie, Media & Telecom",
            topic_keywords=["agentic ai", "cybersecurity"],
            language="en",
        ),
        profiles=ProfileBundle(workflow=(), domain_expert=()),
        snippets=snippets,
    )
    defaults.update(overrides)
    return ContentRequest(**defaults)


def test_stub_path_when_no_api_key():
    result = generate_content(_req(), api_key=None, model=None)
    assert result.source == "deterministic"
    assert result.model is None
    assert "Stub content" in result.body
    # Stub should at least surface the refined prompt for the user
    assert "agentic AI for IT directors" in result.body
    assert result.citations == []


def test_llm_path_returns_parsed_with_validated_citations():
    req = _req()
    fake_parsed = ContentResult(
        body="Agentic AI is reshaping cyber [1]. The TMT sector feels it most [2].",
        citations=[
            Citation(index=1, source_doc="WRONG", node_id="WRONG", title="WRONG"),
            Citation(index=2, source_doc="WRONG", node_id="WRONG", title="WRONG"),
            Citation(index=99, source_doc="ghost", node_id="ghost", title="ghost"),
        ],
        reasoning="written",
    )
    fake_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=fake_parsed))]
    )
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse.return_value = fake_completion

    with patch("backend.content_generation.service.openai.OpenAI", return_value=fake_client):
        result = generate_content(req, api_key="sk-test", model="gpt-4o")

    assert result.source == "llm"
    assert result.model == "gpt-4o"
    assert "[1]" in result.body and "[2]" in result.body
    # Out-of-range citation dropped; remaining ones rewritten from real snippets.
    assert len(result.citations) == 2
    assert result.citations[0].source_doc == "corpus_en"
    assert result.citations[0].node_id == "0114"
    assert result.citations[1].title == "Cybersecurity in TMT"
    # Confirm the patched client was actually used.
    fake_client.beta.chat.completions.parse.assert_called_once()


def test_error_falls_back_to_stub():
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse.side_effect = RuntimeError("boom")

    with patch("backend.content_generation.service.openai.OpenAI", return_value=fake_client):
        result = generate_content(_req(), api_key="sk-test", model="gpt-4o")

    assert result.source == "deterministic"
    assert "Stub content" in result.body
    assert "boom" in result.reasoning


def test_stub_body_includes_snippet_summary():
    result = generate_content(_req(), api_key=None, model=None)
    assert "Agentic AI" in result.body
    assert "corpus_en::0114" in result.body
