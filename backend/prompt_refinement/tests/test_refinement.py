"""Smoke tests for the refinement stage.

Stub generator path (no LLM): verifies advance_turn produces sensible
questions and apply_user_reply mutates the refined prompt as expected.

needs_re_retrieval covers pivot vs non-pivot intent shifts.
"""

from __future__ import annotations

from backend.intent import IntentResult
from backend.prompt_refinement import (
    RefinedPrompt,
    RefinementTurn,
    advance_turn,
    append_qa,
    needs_re_retrieval,
)
from backend.prompt_refinement.service import overwrite_prompt
from backend.retrieval.types import RetrievalQuery, RetrievalResult, Snippet
from profiles import ProfileBundle


def _intent(**overrides) -> IntentResult:
    defaults = dict(
        role="editor",
        task_codes=["T1_DRAFT"],
        confidence=0.9,
        task_reason="testing",
        sector="Technologie, Media & Telecom",
        topic_keywords=["cybersecurity", "agentic AI"],
        language="en",
    )
    defaults.update(overrides)
    return IntentResult(**defaults)


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        snippets=[
            Snippet(
                source_doc="corpus_en",
                node_id="0114",
                title="The two faces of Agentic AI: weapon and shield",
                content="Agentic AI is both attacker and defender…",
                score=1.0,
                reason="title match",
            ),
            Snippet(
                source_doc="corpus_en",
                node_id="0079",
                title="Cybersecurity in the TMT sector",
                content="The TMT sector's cyber posture is…",
                score=0.9,
                reason="title match",
            ),
        ],
        provider="pageindex",
        corpora_searched=["corpus_en"],
        source="deterministic",
    )


def test_stub_generator_returns_questions_with_choices():
    turn, output = advance_turn(
        original_prompt="Write something about cybersecurity",
        refined_prompt="Write something about cybersecurity",
        intent=_intent(language=None),  # missing language → expect language question
        profiles=ProfileBundle(workflow=[], domain_expert=[]),
        retrieval=_retrieval(),
        prior_turns=[],
        api_key=None,
        model=None,
    )
    assert turn.role == "assistant"
    assert output.questions, "stub should produce at least one question"
    # Every stub question must come with click-to-pick choices
    for q in output.questions:
        assert q.question
        assert q.choices, f"stub question must include choices: {q.question}"
    # Language question should appear when language is missing
    assert any(
        "language" in q.question.lower() or "english" in q.question.lower()
        for q in output.questions
    )


def test_append_qa_appends_under_clarifications():
    refined = RefinedPrompt(
        original="Draft a cyber article", refined="Draft a cyber article"
    )
    new = append_qa(refined, "What audience?", "Board-level execs")
    assert new.turns_count == 1
    assert "Clarifications:" in new.refined
    assert "Board-level execs" in new.refined
    # Original prompt preserved at the top
    assert new.refined.startswith("Draft a cyber article")


def test_append_qa_appends_under_existing_block():
    refined = RefinedPrompt(
        original="Draft a cyber article",
        refined="Draft a cyber article\n\nClarifications:\n- What audience?: Board execs",
    )
    new = append_qa(refined, "What length?", "Around 400 words")
    assert new.refined.count("Clarifications:") == 1
    assert "Around 400 words" in new.refined


def test_overwrite_prompt_replaces_verbatim():
    refined = RefinedPrompt(
        original="Draft a cyber article", refined="Draft a cyber article"
    )
    new = overwrite_prompt(refined, "Draft a 500-word cyber briefing for IT directors")
    assert new.refined == "Draft a 500-word cyber briefing for IT directors"


def test_needs_re_retrieval_false_on_identical_intent():
    a = _intent()
    assert needs_re_retrieval(a, a) is False


def test_needs_re_retrieval_true_on_topic_pivot():
    a = _intent(topic_keywords=["cybersecurity"])
    b = _intent(topic_keywords=["5G", "telecom"])
    assert needs_re_retrieval(a, b) is True


def test_needs_re_retrieval_true_on_task_pivot():
    a = _intent(task_codes=["T1_DRAFT"])
    b = _intent(task_codes=["T1_TRANSLATE"])
    assert needs_re_retrieval(a, b) is True


def test_needs_re_retrieval_false_on_minor_keyword_addition():
    a = _intent(topic_keywords=["cybersecurity", "agentic AI"])
    b = _intent(topic_keywords=["cybersecurity", "agentic AI", "workforce shortage"])
    # Jaccard = 2/3 ≈ 0.67 — above 0.5 threshold, no re-retrieval
    assert needs_re_retrieval(a, b) is False
