"""Smoke tests for the public :mod:`scripts` API surface.

These tests verify the contract documented in ``context-engineering/README.md``:
the ``RAG`` class, the ``ContextBundle`` / ``RetrievedChunk`` dataclasses,
prompt composition, and (when indexes are built) end-to-end retrieval.

Run with::

    pip install pytest
    python -m pytest tests/

Tiering:
  - The first block runs on any clone — no indexes or heavy deps required.
  - The end-to-end block auto-skips if the default-recipe ``vector_db/``
    indexes aren't built (run ``scripts.sh`` or the embed commands in the
    README setup to build them).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from scripts import (
    DEFAULT_ARTICLES_CHUNKER,
    DEFAULT_EMBEDDER,
    DEFAULT_TOP_K_ARTICLES,
    DEFAULT_TOP_K_RULES,
    DEFAULT_WG_CHUNKER,
    RAG,
    SYSTEM_PROMPT,
    ContextBundle,
    RetrievedChunk,
    compose_user_message,
)

ROOT = Path(__file__).resolve().parents[1]      # context-engineering/
VECTOR_DB = ROOT / "vector_db"


# ---------------------------------------------------------------------------
# Tier 1 — public surface contract (no deps, no indexes)
# ---------------------------------------------------------------------------


def test_public_imports_resolve() -> None:
    """Every name documented in the README is exported and non-empty."""
    assert RAG is not None
    assert is_dataclass(ContextBundle)
    assert is_dataclass(RetrievedChunk)
    assert SYSTEM_PROMPT.strip()
    assert callable(compose_user_message)

    # Defaults haven't drifted from what the README documents
    assert DEFAULT_EMBEDDER == "e4"
    assert DEFAULT_ARTICLES_CHUNKER == "a9"
    assert DEFAULT_WG_CHUNKER == "a10"
    assert DEFAULT_TOP_K_ARTICLES == 3
    assert DEFAULT_TOP_K_RULES == 5


def test_context_bundle_shape() -> None:
    """ContextBundle exposes exactly the 5 documented fields."""
    bundle = ContextBundle(
        query="test",
        style_references=[],
        writing_rules=[],
        composed_prompt="",
    )
    assert bundle.query == "test"
    assert bundle.style_references == []
    assert bundle.writing_rules == []
    assert bundle.composed_prompt == ""
    assert bundle.debug == {}                          # default_factory=dict

    assert {f.name for f in fields(ContextBundle)} == {
        "query",
        "style_references",
        "writing_rules",
        "composed_prompt",
        "debug",
    }


def test_retrieved_chunk_shape() -> None:
    """RetrievedChunk exposes exactly the 3 documented fields."""
    chunk = RetrievedChunk(text="hello", metadata={"source_title": "X"})
    assert chunk.text == "hello"
    assert chunk.metadata == {"source_title": "X"}
    assert chunk.score == 0.0                          # default

    assert {f.name for f in fields(RetrievedChunk)} == {"text", "metadata", "score"}


def test_rag_constructor_defaults() -> None:
    """RAG() with no args uses the documented defaults."""
    rag = RAG()
    assert rag.embedder == DEFAULT_EMBEDDER
    assert rag.articles_chunker == DEFAULT_ARTICLES_CHUNKER
    assert rag.wg_chunker == DEFAULT_WG_CHUNKER
    assert rag.top_k_articles == DEFAULT_TOP_K_ARTICLES
    assert rag.top_k_rules == DEFAULT_TOP_K_RULES
    assert rag.use_reranker is True
    assert rag.expand_parents is True


def test_rag_constructor_overrides() -> None:
    """Constructor kwargs take effect."""
    rag = RAG(
        embedder="e2",
        articles_chunker="a5",
        top_k_articles=10,
        top_k_rules=7,
        use_reranker=False,
        expand_parents=False,
    )
    assert rag.embedder == "e2"
    assert rag.articles_chunker == "a5"
    assert rag.top_k_articles == 10
    assert rag.top_k_rules == 7
    assert rag.use_reranker is False
    assert rag.expand_parents is False


def test_compose_user_message_with_empty_inputs() -> None:
    """Empty rules/refs produces fallback markers and keeps the closing line."""
    msg = compose_user_message("write something useful", rules=[], references=[])
    assert "WRITING RULES" in msg
    assert "STYLE REFERENCES" in msg
    assert "TASK:" in msg
    assert "write something useful" in msg
    assert "(no rules retrieved)" in msg
    assert "(no references retrieved)" in msg
    assert msg.rstrip().endswith("Write the article now.")


def test_compose_user_message_with_populated_inputs() -> None:
    """Given real chunks, the output surfaces breadcrumb / title / date / sector."""
    rules = [
        RetrievedChunk(
            text="Use British English throughout.",
            metadata={"breadcrumb": "4. Accuracy > 4.1 Spelling > British English"},
        ),
    ]
    refs = [
        RetrievedChunk(
            text="Some article body explaining AI in advertising.",
            metadata={
                "source_title": "AI in Advertising",
                "source_date": "2024-01-01",
                "sector": "TMT",
            },
        ),
    ]
    msg = compose_user_message("the task at hand", rules=rules, references=refs)

    # rule rendering
    assert "4. Accuracy > 4.1 Spelling > British English" in msg
    assert "Use British English throughout." in msg
    # reference rendering
    assert "AI in Advertising" in msg
    assert "2024-01-01" in msg
    assert "sector: TMT" in msg
    assert "Some article body explaining AI in advertising." in msg
    # task pass-through
    assert "the task at hand" in msg


# ---------------------------------------------------------------------------
# Tier 2 — end-to-end retrieval (auto-skips if indexes missing)
# ---------------------------------------------------------------------------


def _indexes_present() -> bool:
    """Return True iff the default-recipe vector_db indexes exist locally."""
    e4_dir = VECTOR_DB / "e4-bge-m3"
    x4_dir = VECTOR_DB / "x4-bm25"
    if not (e4_dir.exists() and x4_dir.exists()):
        return False
    required = [
        e4_dir / "a9-hybrid-small-to-big__gpt-5",
        x4_dir / "a9-hybrid-small-to-big__gpt-5",
        e4_dir / "a10-raptor-structural__writing-guide",
        x4_dir / "a10-raptor-structural__writing-guide",
    ]
    return all(p.exists() for p in required)


_SKIP_REASON = (
    "default-recipe vector_db indexes not built locally — run scripts.sh "
    "or the embed commands in the README setup section"
)


@pytest.mark.skipif(not _indexes_present(), reason=_SKIP_REASON)
def test_retrieve_end_to_end() -> None:
    """RAG().retrieve() on a real query returns a populated ContextBundle."""
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("rank_bm25")

    rag = RAG()
    bundle = rag.retrieve("AI in advertising")

    assert isinstance(bundle, ContextBundle)
    assert bundle.query == "AI in advertising"
    assert isinstance(bundle.style_references, list)
    assert isinstance(bundle.writing_rules, list)
    assert isinstance(bundle.composed_prompt, str) and bundle.composed_prompt
    assert isinstance(bundle.debug, dict)
    assert "retrieval_seconds" in bundle.debug

    # Default recipe asks for top-3 articles and top-5 rules. Be lenient on
    # exact count (limited corpus may return fewer), but require at least one.
    assert 1 <= len(bundle.style_references) <= 3
    assert 1 <= len(bundle.writing_rules) <= 5

    for chunk in bundle.style_references + bundle.writing_rules:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.text
        assert isinstance(chunk.metadata, dict)
        assert isinstance(chunk.score, float)


@pytest.mark.skipif(not _indexes_present(), reason=_SKIP_REASON)
def test_retrieve_metadata_keys() -> None:
    """The metadata keys documented in the README actually appear on hits."""
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("rank_bm25")

    rag = RAG()
    bundle = rag.retrieve("cybersecurity in TMT")

    # Article chunks: source_title + source_date + sector
    if bundle.style_references:
        meta = bundle.style_references[0].metadata
        assert "source_title" in meta
        assert "source_date" in meta
        assert "sector" in meta

    # Writing-guide chunks: breadcrumb (or section_title fallback) + a
    # tree-position marker (depth or node_id)
    if bundle.writing_rules:
        meta = bundle.writing_rules[0].metadata
        assert ("breadcrumb" in meta) or ("section_title" in meta)
        assert ("depth" in meta) or ("node_id" in meta)
