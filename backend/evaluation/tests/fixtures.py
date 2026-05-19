"""Shared test fixtures for the evaluation module."""

from __future__ import annotations

from backend.content_generation.types import Citation, ContentRequest, ContentResult
from backend.intent import IntentResult
from backend.retrieval.types import Snippet
from profiles import ProfileBundle


def make_request(**overrides) -> ContentRequest:
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
        refined_prompt="Write a short briefing on agentic AI for IT directors.",
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


def make_generation(body: str | None = None, citations=None) -> ContentResult:
    return ContentResult(
        body=body
        or (
            "# Agentic AI for IT directors\n\n"
            "Agentic AI is reshaping cyber [1]. It acts as attacker and defender [2].\n\n"
            "- Watch AI agents in red teams\n"
            "- Update incident response\n"
            "- Train staff\n"
        ),
        citations=citations
        if citations is not None
        else [
            Citation(
                index=1,
                source_doc="corpus_en",
                node_id="0114",
                title="The two faces of Agentic AI",
            ),
            Citation(
                index=2,
                source_doc="corpus_en",
                node_id="0079",
                title="Cybersecurity in TMT",
            ),
        ],
        model="gpt-4o",
        source="llm",
        reasoning="ok",
    )
