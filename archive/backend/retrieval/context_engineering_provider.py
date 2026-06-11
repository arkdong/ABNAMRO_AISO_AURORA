"""Track A — Vector RAG provider.

Adapter around ``context-engineering``'s ``scripts.api.RAG``. Translates
between two shapes:

- Track A returns a ``ContextBundle`` split into ``style_references`` and
  ``writing_rules`` (both ``list[RetrievedChunk]`` with rich metadata).
- The demo's retrieval layer expects a single flat ``list[Snippet]`` on
  :class:`backend.retrieval.types.RetrievalResult`.

The underlying ``RAG`` instance is expensive on first query — BGE-M3 +
bge-reranker-v2-m3 load lazily inside ``scripts.retrieve``. Constructing
``RAG`` itself is cheap; we instantiate it on first ``.retrieve()`` call and
cache it on the provider, so a Streamlit ``@st.cache_resource`` wrapper around
the provider amortises the model load across reruns.

A ``sys.path`` shim adds ``<repo_root>/context-engineering`` so
``from scripts import RAG`` resolves — the hyphenated directory name
precludes a normal package import.

If the local ``vector_db/`` is missing the default ``a9``/``a10`` × ``e4``/``x4``
collections, ``.retrieve()`` returns an empty result and logs a warning rather
than crashing the demo pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.retrieval.types import RetrievalQuery, RetrievalResult, Snippet

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CE_DIR = REPO_ROOT / "context-engineering"
CE_VECTOR_DB = CE_DIR / "vector_db"


def _ensure_ce_on_path() -> None:
    """Make ``scripts`` (Track A's package) importable.

    The folder name ``context-engineering`` is hyphenated, so a plain
    ``import context_engineering`` won't work. Instead we put the directory
    itself on ``sys.path`` and import its inner ``scripts`` package directly,
    matching how Track A's own CLI invokes itself (``python -m scripts.generate``).
    """
    if CE_DIR.exists() and str(CE_DIR) not in sys.path:
        sys.path.insert(0, str(CE_DIR))


def _ratio_split(k: int) -> tuple[int, int]:
    """Split ``k`` snippets into (articles, rules) at Track A's 3:5 default ratio.

    ``scripts.api`` defaults to ``top_k_articles=3`` and ``top_k_rules=5``; we
    preserve that mix when the caller passes an arbitrary ``k``. Floors at 1
    each side so a small ``k`` still returns one of each kind.
    """
    if k <= 2:
        return (max(1, k // 2 or 1), max(1, k - max(1, k // 2 or 1)))
    rules = max(1, round(k * 5 / 8))
    articles = max(1, k - rules)
    return (articles, rules)


class ContextEngineeringProvider:
    """Hybrid dense + BM25 + cross-encoder rerank, via ``scripts.api.RAG``."""

    name = "context_engineering"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        use_reranker: bool = True,
        top_k_articles: Optional[int] = None,
        top_k_rules: Optional[int] = None,
    ) -> None:
        # ``OPENAI_API_KEY_VECTOR_RAG`` drives the LLM half of Track A and
        # isn't needed for retrieval-only (BGE-M3 + BM25 run locally). We
        # still surface it under the generic ``OPENAI_API_KEY`` name so any
        # downstream ``RAG.generate`` call finds it without extra wiring.
        key = api_key or os.getenv("OPENAI_API_KEY_VECTOR_RAG")
        if key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = key
        self._use_reranker = use_reranker
        self._explicit_articles = top_k_articles
        self._explicit_rules = top_k_rules
        self._rag = None  # lazy

    # ---- lazy RAG -----------------------------------------------------------

    def _ensure_rag(self):
        if self._rag is not None:
            return self._rag
        _ensure_ce_on_path()
        try:
            from scripts import RAG  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"Cannot import Track A RAG: {e}. Expected the "
                f"context-engineering package at {CE_DIR}."
            ) from e
        logger.info(
            f"ContextEngineering: instantiating RAG (use_reranker={self._use_reranker})"
        )
        self._rag = RAG(use_reranker=self._use_reranker)
        return self._rag

    # ---- adapter ------------------------------------------------------------

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        # Cheap upfront sanity check so a missing build surfaces clearly
        # instead of as a Chroma traceback two layers down.
        if not CE_VECTOR_DB.exists():
            logger.warning(
                f"ContextEngineering: vector_db missing at {CE_VECTOR_DB}; "
                "returning empty result. Build with context-engineering/scripts.sh."
            )
            return RetrievalResult(
                snippets=[],
                provider=self.name,
                corpora_searched=[],
                source="deterministic",
            )

        if self._explicit_articles is not None or self._explicit_rules is not None:
            top_k_articles = self._explicit_articles or 0
            top_k_rules = self._explicit_rules or 0
        else:
            top_k_articles, top_k_rules = _ratio_split(query.k)

        try:
            rag = self._ensure_rag()
            bundle = rag.retrieve(
                query.user_prompt,
                top_k_articles=top_k_articles,
                top_k_rules=top_k_rules,
            )
        except Exception as e:
            # Empty result + warning, never a 500: the orchestrator may still
            # have useful snippets from another provider.
            logger.warning(f"ContextEngineering retrieval failed: {e!r}")
            return RetrievalResult(
                snippets=[],
                provider=self.name,
                corpora_searched=[],
                source="deterministic",
            )

        snippets = _bundle_to_snippets(bundle)
        snippets.sort(key=lambda s: -s.score)
        return RetrievalResult(
            snippets=snippets,
            provider=self.name,
            corpora_searched=["articles_gpt5", "writing_guide"],
            # Retrieval path uses embeddings + BM25 + cross-encoder rerank,
            # no LLM ranker → mark as deterministic for consistency with the
            # PageIndex provider's deterministic-fallback labelling.
            source="deterministic",
        )


# ── Helpers ────────────────────────────────────────────────────────────────


def _bundle_to_snippets(bundle) -> list[Snippet]:
    """Flatten a Track A ``ContextBundle`` into ``Snippet`` rows.

    Field mapping:

    - articles  → ``source_doc=source_slug``, ``title=source_title``,
                  ``reason="style reference (Vector RAG)"``
    - rules     → ``source_doc="writing_guide"``, ``title=breadcrumb``,
                  ``reason="writing rule (Vector RAG)"``
    """
    out: list[Snippet] = []
    for c in bundle.style_references:
        md = c.metadata or {}
        out.append(
            Snippet(
                source_doc=str(md.get("source_slug") or "article"),
                node_id=str(md.get("chunk_id") or md.get("node_id") or len(out)),
                title=str(md.get("source_title") or "Article"),
                content=c.text,
                line_num=None,
                score=float(c.score),
                reason="style reference (Vector RAG)",
            )
        )
    for c in bundle.writing_rules:
        md = c.metadata or {}
        out.append(
            Snippet(
                source_doc="writing_guide",
                node_id=str(md.get("node_id") or md.get("chunk_id") or len(out)),
                title=str(
                    md.get("breadcrumb")
                    or md.get("section_title")
                    or "Writing rule"
                ),
                content=c.text,
                line_num=None,
                score=float(c.score),
                reason="writing rule (Vector RAG)",
            )
        )
    return out
