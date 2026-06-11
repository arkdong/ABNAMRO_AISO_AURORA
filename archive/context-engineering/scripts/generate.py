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

This file is a thin CLI on top of :mod:`scripts.api`. To use the same
pipeline from Python, import ``RAG`` directly:

    from scripts import RAG
    bundle = RAG().retrieve("...")
"""

from __future__ import annotations

import argparse
import os
import sys

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
    RAG,
    call_anthropic,
    call_openai,
    compose_user_message,
)
from .retrieve import EMBEDDER_MODELS


# ---------------------------------------------------------------------------
# Display helpers — CLI presentation only
# ---------------------------------------------------------------------------


def _print_section(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f"  {title}")
    print(f"{char * 78}")


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + ("..." if len(s) > n else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="candidates per retriever before fusion")
    ap.add_argument("--top-k-articles", type=int, default=DEFAULT_TOP_K_ARTICLES)
    ap.add_argument("--top-k-rules", type=int, default=DEFAULT_TOP_K_RULES)
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the cross-encoder reranker (faster, lower quality)")
    ap.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
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

    rag = RAG(
        embedder=args.embedder,
        articles_chunker=args.articles_chunker,
        wg_chunker=args.wg_chunker,
        articles_src=args.articles_src,
        wg_src=args.wg_src,
        top_n=args.top_n,
        top_k_articles=args.top_k_articles,
        top_k_rules=args.top_k_rules,
        use_reranker=not args.no_rerank,
        reranker_model=args.reranker_model,
        expand_parents=args.expand_parents,
    )

    # ---- Stage 1: retrieval (separate calls so progress messages interleave correctly) ----
    print(f"[generate] query: {args.query!r}", file=sys.stderr)
    print(f"[generate] retrieving article style references "
          f"({args.articles_chunker} × {args.articles_src} via {args.embedder} + BM25)...",
          file=sys.stderr)
    references = rag.retrieve_style_references(args.query)

    print(f"[generate] retrieving writing-guide rules "
          f"({args.wg_chunker} × {args.wg_src} via {args.embedder} + BM25 + tree expansion)...",
          file=sys.stderr)
    rules = rag.retrieve_writing_rules(args.query)

    print(f"[generate] composing prompt with {len(references)} style refs and {len(rules)} rules",
          file=sys.stderr)

    # ---- Stage 2: compose prompt ----
    user_message = compose_user_message(args.query, rules, references)

    # ---- Stage 3: display retrieved context ----
    _print_section(f"STYLE REFERENCES ({len(references)} articles)")
    for i, a in enumerate(references, 1):
        m = a.metadata
        print(f"  {i}. {m.get('source_title', '?')[:75]} "
              f"({m.get('source_date', '?')}, sector: {m.get('sector', '?')})")
        if args.show_context:
            print(f"     {_truncate(a.text, 220)}")

    _print_section(f"WRITING RULES ({len(rules)} rules)")
    for i, r in enumerate(rules, 1):
        m = r.metadata
        crumb = m.get("breadcrumb") or m.get("section_title") or "?"
        print(f"  {i}. {crumb[:90]}")
        if args.show_context:
            print(f"     {_truncate(r.text, 220)}")

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
