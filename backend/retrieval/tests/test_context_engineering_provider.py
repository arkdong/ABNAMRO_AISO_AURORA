"""Smoke test for the Track A → demo adapter.

Skipped automatically when the local ``context-engineering/vector_db/`` build
isn't present, so this file is safe to land in CI without requiring every
contributor to run ``scripts.embed`` first.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.retrieval import (
    ContextEngineeringProvider,
    RetrievalQuery,
    Snippet,
)
from backend.retrieval.context_engineering_provider import (
    CE_VECTOR_DB,
    _bundle_to_snippets,
    _ratio_split,
)

REQUIRED_COLLECTIONS = (
    CE_VECTOR_DB / "e4-bge-m3",
    CE_VECTOR_DB / "x4-bm25" / "a9-hybrid-small-to-big__gpt-5",
    CE_VECTOR_DB / "x4-bm25" / "a10-raptor-structural__writing-guide",
)


def _vector_db_built() -> bool:
    return all(p.exists() for p in REQUIRED_COLLECTIONS)


# ---- pure helpers (no model load) -----------------------------------------


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, (1, 1)),   # never empty on either side
        (2, (1, 1)),
        (5, (2, 3)),   # 3:5 ratio, rounded
        (8, (3, 5)),   # matches Track A defaults
        (10, (4, 6)),
    ],
)
def test_ratio_split_floors_at_one(k: int, expected: tuple[int, int]) -> None:
    """``_ratio_split`` keeps both kinds non-empty and mirrors Track A's 3:5 mix."""
    assert _ratio_split(k) == expected


def test_bundle_to_snippets_maps_metadata() -> None:
    """Field mapping: source_title→title for articles, breadcrumb→title for rules."""
    from types import SimpleNamespace

    bundle = SimpleNamespace(
        style_references=[
            SimpleNamespace(
                text="article body",
                score=0.91,
                metadata={
                    "source_slug": "ai-advertising",
                    "source_title": "AI in advertising",
                    "chunk_id": "c1",
                },
            )
        ],
        writing_rules=[
            SimpleNamespace(
                text="rule body",
                score=0.84,
                metadata={
                    "breadcrumb": "3. Wording > 3.1 Style > Reader-first",
                    "node_id": "n12",
                    "chunk_id": "c2",
                },
            )
        ],
    )
    out = _bundle_to_snippets(bundle)
    assert len(out) == 2
    article, rule = out
    assert isinstance(article, Snippet) and isinstance(rule, Snippet)
    assert article.title == "AI in advertising"
    assert article.source_doc == "ai-advertising"
    assert article.reason.startswith("style reference")
    assert rule.title == "3. Wording > 3.1 Style > Reader-first"
    assert rule.source_doc == "writing_guide"
    assert rule.reason.startswith("writing rule")


def test_retrieve_returns_empty_when_vector_db_missing(tmp_path, monkeypatch) -> None:
    """Provider degrades gracefully when no build exists — never raises into the demo."""
    import backend.retrieval.context_engineering_provider as cep

    monkeypatch.setattr(cep, "CE_VECTOR_DB", tmp_path / "missing")
    provider = cep.ContextEngineeringProvider()
    result = provider.retrieve(
        RetrievalQuery(user_prompt="any query", k=5)
    )
    assert result.snippets == []
    assert result.provider == "context_engineering"
    assert result.corpora_searched == []


# ---- end-to-end (needs local vector_db) -----------------------------------


@pytest.mark.skipif(
    not _vector_db_built(),
    reason=(
        "Track A vector_db not built; run context-engineering/scripts.sh "
        "to enable this test."
    ),
)
def test_retrieve_end_to_end_returns_snippets() -> None:
    """First-class smoke: query in, Snippets out, scores monotonically descending."""
    # Restrict to a handful of snippets so reranker time stays bounded.
    provider = ContextEngineeringProvider(use_reranker=False)
    result = provider.retrieve(
        RetrievalQuery(
            user_prompt="How is AI changing advertising for Dutch businesses?",
            k=4,
        )
    )
    assert result.provider == "context_engineering"
    assert result.snippets, "expected at least one snippet from Track A"
    # Snippets are sorted by score desc.
    scores = [s.score for s in result.snippets]
    assert scores == sorted(scores, reverse=True)
    # At least one of the two kinds should be present.
    kinds = {s.reason.split(" ")[0] for s in result.snippets}
    assert kinds & {"style", "writing"}
