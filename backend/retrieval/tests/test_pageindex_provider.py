"""PageIndex provider: corpus routing + deterministic retrieval against the
real cached trees in ``rag/corpus/``.
"""

from __future__ import annotations

from backend.retrieval import PageIndexProvider, RetrievalQuery


def test_corpus_routing_search_only_uses_corpus_en():
    provider = PageIndexProvider()
    q = RetrievalQuery(user_prompt="any related articles?", task_codes=["T1_SEARCH"])
    r = provider.retrieve(q)
    assert r.corpora_searched == ["corpus_en"]


def test_corpus_routing_compliance_unions_writing_guide_and_corpus():
    provider = PageIndexProvider()
    q = RetrievalQuery(user_prompt="check against the guide", task_codes=["T2_COMPLIANCE"])
    r = provider.retrieve(q)
    assert set(r.corpora_searched) == {"writing_guide", "corpus_en"}


def test_corpus_routing_multi_intent_unions():
    provider = PageIndexProvider()
    q = RetrievalQuery(
        user_prompt="vertaal en zoek gerelateerd",
        task_codes=["T1_TRANSLATE", "T1_SEARCH"],
    )
    r = provider.retrieve(q)
    assert r.corpora_searched == ["corpus_en"]


def test_deterministic_finds_cyber_articles_for_query1():
    provider = PageIndexProvider(api_key=None, model=None)
    q = RetrievalQuery(
        user_prompt=(
            "Write a short analysis article in English on how Agentic AI is changing "
            "the cybersecurity arms race for Dutch TMT companies, and what the "
            "workforce-shortage angle means for IT-leveranciers."
        ),
        task_codes=["T1_DRAFT"],
        topic_keywords=["agentic AI", "cybersecurity"],
        sector="Technologie, Media & Telecom",
        k=5,
    )
    r = provider.retrieve(q)
    assert r.source == "deterministic"
    assert r.provider == "pageindex"
    assert len(r.snippets) > 0
    titles = " | ".join(s.title.lower() for s in r.snippets)
    assert any(kw in titles for kw in ("cyber", "agentic", "ai"))


def test_deterministic_finds_agentic_ai_for_query3():
    provider = PageIndexProvider(api_key=None, model=None)
    q = RetrievalQuery(
        user_prompt=(
            "Vertaal het artikel 'The two faces of Agentic AI' naar het Nederlands "
            "en laat me zien welke gerelateerde artikelen we al hebben."
        ),
        task_codes=["T1_TRANSLATE"],
        topic_keywords=["agentic AI"],
        k=5,
    )
    r = provider.retrieve(q)
    assert r.source == "deterministic"
    titles = " | ".join(s.title.lower() for s in r.snippets)
    assert "agentic" in titles
