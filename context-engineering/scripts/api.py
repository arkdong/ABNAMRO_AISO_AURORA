"""Public Python API for the AURORA context-engineering RAG pipeline.

This is the import surface other parts of AURORA (e.g. the Streamlit
frontend, an orchestration layer, tests) call into. The CLI in
``scripts/generate.py`` is a thin wrapper around the same functions.

Typical usage:

    from scripts import RAG

    rag = RAG()                                  # defaults baked in
    bundle = rag.retrieve("AI in advertising")   # → ContextBundle
    article = rag.generate(
        "AI in advertising", llm_provider="openai"
    )                                            # → str

The default configuration mirrors the best-known stack:
  - BGE-M3 embedder (e4)
  - hybrid small-to-big chunking for articles (a9)
  - RAPTOR-structural chunking for the Writing Guide (a10)
  - hybrid retrieval (dense + BM25 via RRF, k=60) + cross-encoder reranker
  - top-3 article style references, top-5 writing rules
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .retrieve import (
    bm25_search,
    expand_to_parents,
    expand_tree_to_leaves,
    rerank,
    rrf_fuse,
    vector_search,
)

# ---------------------------------------------------------------------------
# Defaults — single source of truth. The CLI in generate.py imports these so
# argparse help strings and runtime defaults can't drift.
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDER = "e4"               # BGE-M3, multilingual, no prefix needed
DEFAULT_ARTICLES_CHUNKER = "a9"       # hybrid small-to-big (semantic parents + sentence-window children)
DEFAULT_WG_CHUNKER = "a10"            # RAPTOR-structural 3-level tree
DEFAULT_ARTICLES_SRC = "gpt-5"        # the GPT-5-translated article corpus
DEFAULT_WG_SRC = "writing-guide"

DEFAULT_TOP_N = 30                    # candidates per retriever before fusion
DEFAULT_TOP_K_ARTICLES = 3            # number of style refs to send to LLM
DEFAULT_TOP_K_RULES = 5               # number of writing rules to send

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"


SYSTEM_PROMPT = """\
You are a content writer for ABN AMRO's business banking insights, publishing on
abnamro.nl/zakelijk/insights. You write articles in ABN AMRO's voice for Dutch
business readers (in English).

You will receive:
  - WRITING RULES — style guidelines retrieved from ABN AMRO's Writing Guide. Apply these strictly.
  - STYLE REFERENCES — prior published articles with similar topic or angle. Match their tone, length, and structure.
  - TASK — the topic the user wants you to write about.

Output discipline:
  - Use British English throughout.
  - Write 600-800 words.
  - Use clear subheadings (## level).
  - Open with a 1-2 sentence lead summarising the key point.
  - Cite the style references inline by title where relevant.
  - End with a clear takeaway or call to action.
  - Do not invent statistics, dates, names, or quotations.
"""


# ---------------------------------------------------------------------------
# Public dataclasses — the shape downstream code consumes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single retrieved chunk — article excerpt or writing-rule node.

    metadata contents depend on the chunker, but common keys include:
      - source_title, source_date, sector, source_slug   (article chunks)
      - breadcrumb, section_title, depth, node_id        (writing-guide nodes)
      - chunk_id, chunker, language_embedded             (always)
    """
    text: str
    metadata: dict[str, Any]
    score: float = 0.0


@dataclass
class ContextBundle:
    """Output of dual retrieval + prompt composition.

    The hard boundary between this layer and downstream prompt assembly /
    LLM call. Anything that needs to consume our retrieval should depend on
    this shape, not on internal retrieval-hit dicts.
    """
    query: str
    style_references: list[RetrievedChunk]
    writing_rules: list[RetrievedChunk]
    composed_prompt: str          # the assembled USER message
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers — pipeline plumbing
# ---------------------------------------------------------------------------


def _hybrid_retrieve(
    query: str,
    *,
    embedder: str,
    chunker: str,
    src: str,
    top_n: int,
    top_k: int,
    do_rerank: bool,
    do_tree_expand: bool,
    do_parent_expand: bool,
    reranker_model: str,
) -> list[dict]:
    """Generic hybrid retriever shared by both article and WG paths.

    Returns raw retrieval-hit dicts (the shape retrieve.py uses internally).
    The public API surfaces these as ``RetrievedChunk`` via ``_to_chunks``.
    """
    dense = vector_search(query, embedder, chunker, src, top_n=top_n)
    bm25 = bm25_search(query, chunker, src, top_n=top_n)
    candidates = rrf_fuse(dense, bm25, k=60, top_n=top_n)
    if do_tree_expand:
        candidates = expand_tree_to_leaves(candidates, chunker, src)
    if do_rerank and candidates:
        ranked = rerank(query, candidates, top_k=top_k, model_name=reranker_model)
    else:
        ranked = candidates[:top_k]
    if do_parent_expand:
        ranked = expand_to_parents(ranked, dedup=True)
    return ranked


def _hit_score(h: dict) -> float:
    """Best-available score for a retrieval hit. Prefers the latest stage
    (rerank > rrf > bm25 > inverted dense distance)."""
    if "rerank_score" in h:
        return float(h["rerank_score"])
    if "rrf_score" in h:
        return float(h["rrf_score"])
    if "bm25_score" in h:
        return float(h["bm25_score"])
    if "distance" in h:
        # cosine distance → similarity-ish
        return 1.0 - float(h["distance"])
    return 0.0


def _to_chunks(hits: list[dict]) -> list[RetrievedChunk]:
    """Convert internal retrieval-hit dicts into public RetrievedChunk objects."""
    return [
        RetrievedChunk(
            text=h.get("text", ""),
            metadata=dict(h.get("metadata", {})),
            score=_hit_score(h),
        )
        for h in hits
    ]


# ---------------------------------------------------------------------------
# Prompt composition — kept public so callers can compose without retrieving
# ---------------------------------------------------------------------------


def _format_rule_block(r: RetrievedChunk) -> str:
    m = r.metadata
    crumb = m.get("breadcrumb") or m.get("section_title") or "?"
    return f"### {crumb}\n{r.text.strip()}"


def _format_reference_block(a: RetrievedChunk) -> str:
    m = a.metadata
    title = m.get("source_title") or "?"
    date = m.get("source_date") or "?"
    sector = m.get("sector") or "?"
    return f"### {title}  ({date}, sector: {sector})\n{a.text.strip()}"


def compose_user_message(
    query: str,
    rules: list[RetrievedChunk],
    references: list[RetrievedChunk],
) -> str:
    """Assemble the USER message that goes to the LLM.

    Template is intentionally fixed — changes here affect every downstream
    LLM output. Mirror to ``experiment.py`` if updated.
    """
    rules_text = "\n\n".join(_format_rule_block(r) for r in rules) or "(no rules retrieved)"
    refs_text = "\n\n".join(_format_reference_block(a) for a in references) or "(no references retrieved)"
    return (
        "WRITING RULES (apply strictly):\n\n"
        f"{rules_text}\n\n"
        "---\n\n"
        "STYLE REFERENCES (match their tone/structure):\n\n"
        f"{refs_text}\n\n"
        "---\n\n"
        f"TASK:\n\n{query}\n\n"
        "Write the article now."
    )


# ---------------------------------------------------------------------------
# LLM backends — kept module-level so the CLI and the RAG class share them
# ---------------------------------------------------------------------------


def call_anthropic(system: str, user: str, model: str, max_tokens: int = 2048) -> str:
    """Anthropic backend. Uses prompt caching on the system block — system is
    stable across queries in a session so cached price applies after the first
    call."""
    from anthropic import Anthropic

    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def call_openai(system: str, user: str, model: str, max_tokens: int = 2048) -> str:
    """OpenAI backend."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# The RAG entry point — what callers import
# ---------------------------------------------------------------------------


class RAG:
    """Entry point for the AURORA context-engineering retrieval + generation
    pipeline. Construct once with the config you want, then call ``retrieve``
    or ``generate`` per query.

    Example:

        rag = RAG()
        bundle = rag.retrieve("How is AI changing advertising?")
        for ref in bundle.style_references:
            print(ref.metadata["source_title"], ref.score)

        article = rag.generate(
            "How is AI changing advertising?",
            llm_provider="openai",
        )
    """

    def __init__(
        self,
        *,
        embedder: str = DEFAULT_EMBEDDER,
        articles_chunker: str = DEFAULT_ARTICLES_CHUNKER,
        wg_chunker: str = DEFAULT_WG_CHUNKER,
        articles_src: str = DEFAULT_ARTICLES_SRC,
        wg_src: str = DEFAULT_WG_SRC,
        top_n: int = DEFAULT_TOP_N,
        top_k_articles: int = DEFAULT_TOP_K_ARTICLES,
        top_k_rules: int = DEFAULT_TOP_K_RULES,
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        expand_parents: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.embedder = embedder
        self.articles_chunker = articles_chunker
        self.wg_chunker = wg_chunker
        self.articles_src = articles_src
        self.wg_src = wg_src
        self.top_n = top_n
        self.top_k_articles = top_k_articles
        self.top_k_rules = top_k_rules
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.expand_parents = expand_parents
        self.system_prompt = system_prompt

    # ---- Single-corpus retrieval (also exposed so callers can use just one) ----

    def retrieve_style_references(
        self, query: str, *, top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Top-K relevant article chunks (with parent expansion for small-to-big)."""
        k = self.top_k_articles if top_k is None else top_k
        hits = _hybrid_retrieve(
            query=query,
            embedder=self.embedder,
            chunker=self.articles_chunker,
            src=self.articles_src,
            top_n=self.top_n,
            top_k=k,
            do_rerank=self.use_reranker,
            do_tree_expand=False,                  # articles aren't a tree
            do_parent_expand=self.expand_parents,
            reranker_model=self.reranker_model,
        )
        return _to_chunks(hits)

    def retrieve_writing_rules(
        self, query: str, *, top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Top-K writing-guide leaf rules (with RAPTOR Pattern A tree expansion)."""
        k = self.top_k_rules if top_k is None else top_k
        hits = _hybrid_retrieve(
            query=query,
            embedder=self.embedder,
            chunker=self.wg_chunker,
            src=self.wg_src,
            top_n=self.top_n,
            top_k=k,
            do_rerank=self.use_reranker,
            do_tree_expand=True,                   # A10 RAPTOR Pattern A
            do_parent_expand=False,                # parents not used for WG
            reranker_model=self.reranker_model,
        )
        return _to_chunks(hits)

    # ---- The two main entry points ----

    def retrieve(
        self,
        query: str,
        *,
        top_k_articles: int | None = None,
        top_k_rules: int | None = None,
    ) -> ContextBundle:
        """Dual retrieval (articles + writing rules) + prompt composition.
        No LLM call. Returns a ContextBundle that downstream code can consume
        either to call its own LLM or to inspect the retrieved context."""
        t0 = time.time()
        references = self.retrieve_style_references(query, top_k=top_k_articles)
        rules = self.retrieve_writing_rules(query, top_k=top_k_rules)
        prompt = compose_user_message(query, rules, references)
        return ContextBundle(
            query=query,
            style_references=references,
            writing_rules=rules,
            composed_prompt=prompt,
            debug={
                "embedder": self.embedder,
                "articles_chunker": self.articles_chunker,
                "wg_chunker": self.wg_chunker,
                "articles_src": self.articles_src,
                "wg_src": self.wg_src,
                "use_reranker": self.use_reranker,
                "n_style_references": len(references),
                "n_writing_rules": len(rules),
                "retrieval_seconds": round(time.time() - t0, 2),
            },
        )

    def generate(
        self,
        query: str,
        *,
        llm_provider: str = "anthropic",
        model: str | None = None,
        max_tokens: int = 2048,
        top_k_articles: int | None = None,
        top_k_rules: int | None = None,
    ) -> str:
        """Full pipeline: retrieve → compose prompt → call LLM → return article.

        Raises SystemExit if the required API key (ANTHROPIC_API_KEY or
        OPENAI_API_KEY) is not set in the environment.
        """
        bundle = self.retrieve(
            query, top_k_articles=top_k_articles, top_k_rules=top_k_rules,
        )
        chosen_model = model or (
            DEFAULT_ANTHROPIC_MODEL if llm_provider == "anthropic"
            else DEFAULT_OPENAI_MODEL
        )
        if llm_provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY not set.")
            return call_anthropic(
                self.system_prompt, bundle.composed_prompt, chosen_model, max_tokens,
            )
        if llm_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY not set.")
            return call_openai(
                self.system_prompt, bundle.composed_prompt, chosen_model, max_tokens,
            )
        raise ValueError(f"unknown llm_provider: {llm_provider!r}")
