"""Retrieval — backend module.

Public surface:

- :class:`RetrievalQuery`, :class:`Snippet`, :class:`RetrievalResult` — data shapes.
- :class:`RagProvider` — protocol any retrieval backend implements.
- :class:`PageIndexProvider` — Track B provider (cached PageIndex trees).
- :class:`ContextEngineeringProvider` — Track A provider (vector + BM25 + rerank).
- :func:`build_query` — assembles a :class:`RetrievalQuery` from upstream stages.
- :func:`retrieve` — orchestrator; fans across providers and merges.
"""

from __future__ import annotations

from backend.retrieval.context_engineering_provider import ContextEngineeringProvider
from backend.retrieval.pageindex_provider import PageIndexProvider
from backend.retrieval.provider import RagProvider
from backend.retrieval.service import build_query, retrieve
from backend.retrieval.types import RetrievalQuery, RetrievalResult, Snippet

__all__ = [
    "ContextEngineeringProvider",
    "PageIndexProvider",
    "RagProvider",
    "RetrievalQuery",
    "RetrievalResult",
    "Snippet",
    "build_query",
    "retrieve",
]
