"""End-to-end RAG generation: query → dual retrieval → LLM → article.

Pipeline:
  1. Hybrid retrieval over articles  (A9 + E4 + X4 + reranker + parent-expand)
       → top-K style references (similar prior articles)
  2. Hybrid retrieval over WG       (A10 + E4 + X4 + tree-expand + reranker)
       → top-K writing rules (specific style guidelines)
  3. Compose structured prompt:
       SYSTEM  — role + always-on instructions
       USER    — rules + style references + the user's task
  4. Call LLM (Anthropic or OpenAI). Or --dry-run to print the prompt.

Usage:
    # Dry-run — no API key needed, prints composed prompt + retrieved context
    python -m scripts.generate --query "I want to write about AI in advertising" --dry-run

    # Real generation (Anthropic, default Haiku)
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m scripts.generate --query "..." --llm-provider anthropic

    # OpenAI / GPT-4o
    export OPENAI_API_KEY=sk-...
    python -m scripts.generate --query "..." --llm-provider openai --model gpt-4o

    # Inspect what was retrieved + the full prompt that was sent
    python -m scripts.generate --query "..." --show-context --show-prompt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .retrieve import (
    EMBEDDER_MODELS,
    bm25_search,
    expand_to_parents,
    expand_tree_to_leaves,
    rerank,
    rrf_fuse,
    vector_search,
)

# ---------------------------------------------------------------------------
# Defaults — wire to the best-known stack we've built so far
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDER = "e4"               # BGE-M3, multilingual, no prefix needed
DEFAULT_ARTICLES_CHUNKER = "a9"       # hybrid small-to-big (semantic parents + sentence-window children)
DEFAULT_WG_CHUNKER = "a10"            # RAPTOR-structural 3-level tree
DEFAULT_ARTICLES_SRC = "gpt-5"        # the GPT-5-translated article corpus
DEFAULT_WG_SRC = "writing-guide"

DEFAULT_TOP_K_ARTICLES = 3            # number of style refs to send to LLM
DEFAULT_TOP_K_RULES = 5               # number of writing rules to send

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
# Retrieval — wraps the hybrid+rerank+expansion pipeline for each source
# ---------------------------------------------------------------------------


def _hybrid_retrieve(
    query: str,
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
    """Generic hybrid retriever shared by both article and WG paths."""
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


def retrieve_style_references(query: str, args) -> list[dict]:
    """Top-K relevant article chunks (with parent expansion)."""
    return _hybrid_retrieve(
        query=query,
        embedder=args.embedder,
        chunker=args.articles_chunker,
        src=args.articles_src,
        top_n=args.top_n,
        top_k=args.top_k_articles,
        do_rerank=not args.no_rerank,
        do_tree_expand=False,                 # articles aren't a tree
        do_parent_expand=args.expand_parents, # A5/A9 small-to-big
        reranker_model=args.reranker_model,
    )


def retrieve_writing_rules(query: str, args) -> list[dict]:
    """Top-K writing-guide leaf rules (with Pattern A tree expansion)."""
    return _hybrid_retrieve(
        query=query,
        embedder=args.embedder,
        chunker=args.wg_chunker,
        src=args.wg_src,
        top_n=args.top_n,
        top_k=args.top_k_rules,
        do_rerank=not args.no_rerank,
        do_tree_expand=True,                  # A10 RAPTOR Pattern A
        do_parent_expand=False,               # parents not used for WG
        reranker_model=args.reranker_model,
    )


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------


def _format_rule_block(r: dict) -> str:
    m = r["metadata"]
    crumb = m.get("breadcrumb") or m.get("section_title") or "?"
    return f"### {crumb}\n{r['text'].strip()}"


def _format_reference_block(a: dict) -> str:
    m = a["metadata"]
    title = m.get("source_title") or "?"
    date = m.get("source_date") or "?"
    sector = m.get("sector") or "?"
    return f"### {title}  ({date}, sector: {sector})\n{a['text'].strip()}"


def compose_user_message(query: str, rules: list[dict], references: list[dict]) -> str:
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
# LLM backends
# ---------------------------------------------------------------------------


def call_anthropic(system: str, user: str, model: str, max_tokens: int = 2048) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            # Anthropic prompt caching on the system prompt — system is stable
            # across queries so subsequent calls in the same session pay cached price.
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def call_openai(system: str, user: str, model: str, max_tokens: int = 2048) -> str:
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
# CLI
# ---------------------------------------------------------------------------


def _print_section(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f"  {title}")
    print(f"{char * 78}")


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + ("..." if len(s) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="what the user wants the LLM to write about")

    # Retrieval config
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER, choices=sorted(EMBEDDER_MODELS),
                    help="dense embedder used for both article and WG retrieval")
    ap.add_argument("--articles-chunker", default=DEFAULT_ARTICLES_CHUNKER)
    ap.add_argument("--articles-src", default=DEFAULT_ARTICLES_SRC,
                    help="translation method folder (gpt-5, deep-translator, source-nl)")
    ap.add_argument("--wg-chunker", default=DEFAULT_WG_CHUNKER)
    ap.add_argument("--wg-src", default=DEFAULT_WG_SRC)
    ap.add_argument("--top-n", type=int, default=30, help="candidates per retriever before fusion")
    ap.add_argument("--top-k-articles", type=int, default=DEFAULT_TOP_K_ARTICLES)
    ap.add_argument("--top-k-rules", type=int, default=DEFAULT_TOP_K_RULES)
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the cross-encoder reranker (faster, lower quality)")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--expand-parents", action=argparse.BooleanOptionalAction, default=True,
                    help="for A5/A9 article chunkers, return parent_text instead of child_text")

    # Generation config
    ap.add_argument("--dry-run", action="store_true",
                    help="don't call LLM; just print composed prompt + retrieved context")
    ap.add_argument("--llm-provider", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--model", default=None,
                    help="override default model (default: claude-haiku-4-5 for anthropic, gpt-4o for openai)")
    ap.add_argument("--max-tokens", type=int, default=2048)

    # Output
    ap.add_argument("--show-context", action="store_true",
                    help="print the actual retrieved chunks (titles only by default)")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the full user message that's sent to the LLM")

    args = ap.parse_args()

    # ---- Stage 1: retrieval ----
    print(f"[generate] query: {args.query!r}", file=sys.stderr)
    print(f"[generate] retrieving article style references "
          f"({args.articles_chunker} × {args.articles_src} via {args.embedder} + BM25)...",
          file=sys.stderr)
    references = retrieve_style_references(args.query, args)

    print(f"[generate] retrieving writing-guide rules "
          f"({args.wg_chunker} × {args.wg_src} via {args.embedder} + BM25 + tree expansion)...",
          file=sys.stderr)
    rules = retrieve_writing_rules(args.query, args)

    print(f"[generate] composing prompt with {len(references)} style refs and {len(rules)} rules",
          file=sys.stderr)

    # ---- Stage 2: compose prompt ----
    user_message = compose_user_message(args.query, rules, references)

    # ---- Stage 3: display retrieved context ----
    _print_section(f"STYLE REFERENCES ({len(references)} articles)")
    for i, a in enumerate(references, 1):
        m = a["metadata"]
        print(f"  {i}. {m.get('source_title', '?')[:75]} "
              f"({m.get('source_date', '?')}, sector: {m.get('sector', '?')})")
        if args.show_context:
            print(f"     {_truncate(a['text'], 220)}")

    _print_section(f"WRITING RULES ({len(rules)} rules)")
    for i, r in enumerate(rules, 1):
        m = r["metadata"]
        crumb = m.get("breadcrumb") or m.get("section_title") or "?"
        print(f"  {i}. {crumb[:90]}")
        if args.show_context:
            print(f"     {_truncate(r['text'], 220)}")

    if args.show_prompt:
        _print_section("FULL USER MESSAGE (what the LLM sees)", "-")
        print(user_message)

    # ---- Stage 4: call LLM (or stop here in dry-run) ----
    if args.dry_run:
        _print_section("DRY-RUN — skipping LLM call")
        n_chars = len(SYSTEM_PROMPT) + len(user_message)
        print(f"\nWould send: SYSTEM={len(SYSTEM_PROMPT)} chars  USER={len(user_message)} chars")
        print(f"Total: {n_chars:,} chars (~{n_chars // 4:,} tokens)")
        print("\nUse --show-prompt to see the full user message.")
        return

    provider_model = (args.model or
                      (DEFAULT_ANTHROPIC_MODEL if args.llm_provider == "anthropic" else DEFAULT_OPENAI_MODEL))
    _print_section(f"CALLING {args.llm_provider.upper()} ({provider_model})")
    if args.llm_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set. Use --dry-run to see the prompt without calling.")
        output = call_anthropic(SYSTEM_PROMPT, user_message, provider_model, args.max_tokens)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set. Use --dry-run to see the prompt without calling.")
        output = call_openai(SYSTEM_PROMPT, user_message, provider_model, args.max_tokens)

    _print_section("GENERATED ARTICLE")
    print(output)
    print()


if __name__ == "__main__":
    main()
