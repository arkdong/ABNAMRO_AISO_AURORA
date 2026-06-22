from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from aurora_tool_server.core import AuroraConfig, AuroraCore
from aurora_tool_server.generation import generate_draft
from aurora_tool_server.intent import classify_intent
from aurora_tool_server.refinement import build_questions
from aurora_tool_server.schemas import ProfileBundleResult, StageOptions
from tests.eval_fixtures import make_intent, make_snippets


def _core() -> AuroraCore:
    return AuroraCore(
        AuroraConfig(
            intent_api_key=None,
            content_api_key=None,
            evaluation_api_key=None,
            retrieval_k=4,
        )
    )


def test_dutch_prompt_detects_dutch_language() -> None:
    intent = _core().classify_intent(
        "Schrijf een artikel over cyberveiligheid in de TMT-sector."
    )

    assert intent.language == "nl"
    assert intent.task_codes == ["T1_DRAFT"]
    assert intent.sector == "Technologie, Media & Telecom"
    assert "cyberveiligheid" in intent.topic_keywords


def test_final_draft_language_instruction_overrides_earlier_language_hint() -> None:
    core = _core()
    prompt = (
        "Write a short analysis article in English on how Agentic AI is changing "
        "the cybersecurity arms race for Dutch TMT companies, and what the "
        "workforce-shortage angle means for IT-leveranciers. Make sure the final "
        "Draft is in dutch"
    )

    intent = core.classify_intent(prompt)
    profiles = core.select_profiles(intent)
    retrieval = core.retrieve_context(prompt, intent, profiles)

    assert intent.language == "nl"
    assert retrieval.corpora_searched[:2] == ["corpus_nl", "schrijfwijzer"]


def test_dutch_prompt_with_english_output_keeps_refinement_questions_dutch() -> None:
    core = _core()
    prompt = (
        "Schrijf een kort analyseartikel in het Engels over hoe Agentic AI de "
        "cybersecurity-wapenwedloop voor Nederlandse TMT-bedrijven verandert, "
        "en wat het tekort aan arbeidskrachten betekent voor IT-leveranciers."
    )

    intent = core.classify_intent(prompt)
    profiles = core.select_profiles(intent)
    retrieval = core.retrieve_context(prompt, intent, profiles)
    refinement = core.refine_prompt(
        prompt,
        intent,
        profiles,
        retrieval,
        answers={},
        ask_questions=True,
    )

    assert intent.language == "en"
    assert retrieval.corpora_searched[:2] == ["corpus_en", "writing_guide"]
    assert refinement.questions
    assert refinement.questions[0].question == "Voor wie is de tekst bedoeld?"


class _ConversationLanguageCompletions:
    messages = None

    def parse(self, *, messages, response_format, **_kwargs):
        type(self).messages = messages
        parsed = response_format(
            questions=[
                {
                    "question": "Voor wie is de tekst bedoeld?",
                    "choices": ["Zakelijke beslissers", "IT-directeuren"],
                }
            ],
            proposed_prompt=None,
            done=False,
            reasoning="Asked in the conversation language.",
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


class _ConversationLanguageOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        completions = _ConversationLanguageCompletions()
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_llm_refinement_receives_prompt_language_separate_from_output_language(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_ConversationLanguageOpenAI))
    core = AuroraCore(
        AuroraConfig(
            intent_api_key=None,
            refinement_api_key="sk-test",
            refinement_model="gpt-test",
            content_api_key=None,
            evaluation_api_key=None,
            retrieval_k=4,
        )
    )
    prompt = (
        "Schrijf een kort analyseartikel in het Engels over hoe Agentic AI de "
        "cybersecurity-wapenwedloop voor Nederlandse TMT-bedrijven verandert."
    )

    intent = core.classify_intent(prompt)
    profiles = core.select_profiles(intent)
    retrieval = core.retrieve_context(prompt, intent, profiles)
    refinement = core.refine_prompt(
        prompt,
        intent,
        profiles,
        retrieval,
        answers={},
        ask_questions=True,
    )

    assert intent.language == "en"
    assert refinement.source == "llm"
    assert refinement.questions[0].question == "Voor wie is de tekst bedoeld?"
    messages = _ConversationLanguageCompletions.messages
    assert messages is not None
    user_message = messages[1]["content"]
    assert "Conversation language: nl" in user_message
    assert '"language": "en"' in user_message


class _LanguageClassifyingCompletions:
    def create(self, *, messages, **_kwargs):
        content = {
            "role": "Insights Editorial",
            "task_codes": ["T1_DRAFT"],
            "confidence": 0.94,
            "task_reason": "The request asks for a short analysis article.",
            "sector": "Technologie, Media & Telecom",
            "topic_keywords": ["agentic AI", "cybersecurity"],
            "language": "en",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )


class _LanguageClassifyingOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_LanguageClassifyingCompletions())


def test_llm_intent_language_honors_final_draft_override(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_LanguageClassifyingOpenAI))
    prompt = (
        "Write a short analysis article in English on agentic AI for Dutch TMT "
        "companies. Make sure the final draft is in Dutch."
    )

    intent = classify_intent(prompt, api_key="sk-test", model="gpt-test")

    assert intent.source == "llm"
    assert intent.language == "nl"


def test_output_language_override_routes_to_dutch_pageindex_assets() -> None:
    core = _core()
    options = StageOptions(output_language="nl", retrieval_backend="pageindex", k=4)
    prompt = "Write an article about cybersecurity in the TMT sector."

    intent = core.classify_intent(prompt, options=options)
    profiles = core.select_profiles(intent, options=options)
    retrieval = core.retrieve_context(prompt, intent, profiles, options=options)

    assert intent.language == "nl"
    assert retrieval.corpora_searched[:2] == ["corpus_nl", "schrijfwijzer"]
    assert retrieval.snippets
    assert {snippet.source_doc for snippet in retrieval.snippets}.issubset(
        {"corpus_nl", "schrijfwijzer", "insights_stijlgids_nl"}
    )


def test_pageindex_routes_insights_style_guide_assets() -> None:
    core = _core()
    english_options = StageOptions(output_language="en", retrieval_backend="pageindex", k=8)
    english_prompt = (
        "Review this Insights article checklist for title, intro, SEO, "
        "Local Focus charts and header image."
    )

    english_intent = core.classify_intent(english_prompt, options=english_options)
    english_profiles = core.select_profiles(english_intent, options=english_options)
    english_retrieval = core.retrieve_context(
        english_prompt,
        english_intent,
        english_profiles,
        options=english_options,
    )

    assert "insights_stijlgids_en" in english_retrieval.corpora_searched
    assert any(
        snippet.source_doc == "insights_stijlgids_en"
        for snippet in english_retrieval.snippets
    )

    dutch_options = StageOptions(output_language="nl", retrieval_backend="pageindex", k=8)
    dutch_prompt = (
        "Controleer een Insights artikel op kop, intro, SEO, "
        "Local Focus grafiek en header beeld."
    )

    dutch_intent = core.classify_intent(dutch_prompt, options=dutch_options)
    dutch_profiles = core.select_profiles(dutch_intent, options=dutch_options)
    dutch_retrieval = core.retrieve_context(
        dutch_prompt,
        dutch_intent,
        dutch_profiles,
        options=dutch_options,
    )

    assert "insights_stijlgids_nl" in dutch_retrieval.corpora_searched
    assert any(
        snippet.source_doc == "insights_stijlgids_nl"
        for snippet in dutch_retrieval.snippets
    )


def test_vector_rag_uses_generated_dutch_assets() -> None:
    core = _core()
    options = StageOptions(output_language="nl", retrieval_backend="vector_rag", k=4)
    prompt = "Schrijf over AI-agents en softwarebedrijven."

    intent = core.classify_intent(prompt, options=options)
    profiles = core.select_profiles(intent, options=options)
    retrieval = core.retrieve_context(prompt, intent, profiles, options=options)

    assert retrieval.provider == "vector_rag"
    assert retrieval.corpora_searched[:2] == ["corpus_nl", "schrijfwijzer"]
    assert retrieval.snippets
    assert any(snippet.source_doc == "corpus_nl" for snippet in retrieval.snippets)
    assert all("vector overlap" in snippet.reason for snippet in retrieval.snippets)


def test_deterministic_generation_stub_is_dutch_for_dutch_intent() -> None:
    result = generate_draft(
        refined_prompt="Schrijf een korte analyse over AI-agents.",
        intent=make_intent(language="nl", topic_keywords=["AI-agents"]),
        profiles=ProfileBundleResult(),
        snippets=make_snippets(),
    )

    assert result.source == "deterministic"
    assert "AURORA onderbouwde concepttekst" in result.body
    assert "Verfijnde instructie" in result.body
    assert "Schrijfwijzer" in result.body


class _CapturingCompletions:
    messages = None

    def create(self, *, messages, **_kwargs):
        type(self).messages = messages
        content = {
            "body": "Nederlandse concepttekst met bron [1].",
            "reasoning": "generated in Dutch",
            "citation_indices": [1],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )


class _CapturingOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_CapturingCompletions())


def test_llm_generation_prompt_enforces_dutch_rules(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_CapturingOpenAI))

    result = generate_draft(
        refined_prompt="Schrijf een korte analyse over AI-agents.",
        intent=make_intent(language="nl", topic_keywords=["AI-agents"]),
        profiles=ProfileBundleResult(),
        snippets=make_snippets(),
        api_key="sk-test",
        model="gpt-test",
    )

    assert result.source == "llm"
    messages = _CapturingCompletions.messages
    assert messages is not None
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "Output language: Dutch" in system_prompt
    assert "Schrijfwijzer" in system_prompt
    assert "formal u-form" in system_prompt
    assert "Output language: Dutch" in user_prompt


def test_refinement_questions_use_dutch_when_intent_is_dutch() -> None:
    questions = build_questions(
        make_intent(language="nl", topic_keywords=["AI-agents"]),
        retrieval=None,
    )

    assert questions
    assert questions[0].question == "Voor wie is de tekst bedoeld?"
    assert questions[0].choices == ["Zakelijke beslissers", "IT-directeuren", "Algemene klanten"]
