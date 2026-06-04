"""Orchestrator: single-provider passthrough + multi-provider merge."""

from __future__ import annotations

from backend.retrieval import RetrievalQuery, RetrievalResult, Snippet, retrieve


class _StubProvider:
    def __init__(self, name: str, snippets, corpora=()):
        self.name = name
        self._snippets = snippets
        self._corpora = list(corpora)

    def retrieve(self, query):
        return RetrievalResult(
            snippets=self._snippets,
            provider=self.name,
            corpora_searched=self._corpora,
            source="deterministic",
        )


def _snip(title: str, score: float) -> Snippet:
    return Snippet(
        source_doc="x",
        node_id="n",
        title=title,
        content="...",
        line_num=None,
        score=score,
        reason="test",
    )


def test_orchestrator_single_provider_passthrough():
    p = _StubProvider("alpha", [_snip("a", 0.9)], corpora=["x"])
    q = RetrievalQuery(user_prompt="test", k=3)
    r = retrieve(q, providers=[p])
    assert r.provider == "alpha"
    assert len(r.snippets) == 1
    assert r.corpora_searched == ["x"]


def test_orchestrator_multi_provider_merges_by_score_and_caps_k():
    p1 = _StubProvider("alpha", [_snip("a1", 0.9), _snip("a2", 0.5)], corpora=["x"])
    p2 = _StubProvider("beta", [_snip("b1", 0.8), _snip("b2", 0.6)], corpora=["y"])
    q = RetrievalQuery(user_prompt="test", k=3)
    r = retrieve(q, providers=[p1, p2])
    assert r.provider == "alpha+beta"
    assert len(r.snippets) == 3
    assert r.snippets[0].title == "a1"
    assert set(r.corpora_searched) == {"x", "y"}
