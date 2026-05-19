"""AURORA context-engineering — public Python API.

Re-exports the stable surface from :mod:`scripts.api` so callers can do::

    from scripts import RAG, ContextBundle, RetrievedChunk

The CLI in :mod:`scripts.generate` is a thin wrapper around the same API.
"""

from .api import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ARTICLES_CHUNKER,
    DEFAULT_ARTICLES_SRC,
    DEFAULT_EMBEDDER,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOP_K_ARTICLES,
    DEFAULT_TOP_K_RULES,
    DEFAULT_TOP_N,
    DEFAULT_WG_CHUNKER,
    DEFAULT_WG_SRC,
    SYSTEM_PROMPT,
    ContextBundle,
    RAG,
    RetrievedChunk,
    call_anthropic,
    call_openai,
    compose_user_message,
)

__all__ = [
    "RAG",
    "ContextBundle",
    "RetrievedChunk",
    "SYSTEM_PROMPT",
    "compose_user_message",
    "call_anthropic",
    "call_openai",
    "DEFAULT_EMBEDDER",
    "DEFAULT_ARTICLES_CHUNKER",
    "DEFAULT_WG_CHUNKER",
    "DEFAULT_ARTICLES_SRC",
    "DEFAULT_WG_SRC",
    "DEFAULT_TOP_N",
    "DEFAULT_TOP_K_ARTICLES",
    "DEFAULT_TOP_K_RULES",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
]
