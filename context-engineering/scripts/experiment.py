"""Experiment runner: swap prompts & retrieval params, compare results.

This file is intentionally self-contained — the entire experiment workflow
lives here. You edit a few constants near the top, then run:

    python -m scripts.experiment                 # run all experiments
    python -m scripts.experiment --only a,b      # run a subset
    python -m scripts.experiment --dry-run       # skip LLM calls
    python -m scripts.experiment --list          # show defined experiments
    python -m scripts.experiment --list-runs     # show past runs
    python -m scripts.experiment --compare latest   # comparison table

Each run writes one directory per experiment under
    context-engineering/experiments/<run_id>/<experiment_name>/
        config.json                — exact knobs used
        retrieved_references.json  — article chunks the LLM saw
        retrieved_rules.json       — writing-guide chunks the LLM saw
        system_prompt.txt          — system message sent
        user_message.txt           — combined user prompt sent
        output.md                  — LLM output (skipped on --dry-run)
        summary.json               — char/token counts, timings

So diffing two experiments is: diff their user_message.txt / output.md /
retrieved_*.json files. `--compare` prints a one-line-per-experiment table.

----------------------------------------------------------------------------
Which knobs can be swapped, and which can't?
----------------------------------------------------------------------------
Supported (variants provided below):
    top_k_articles    — how many style refs to inject
    top_k_rules       — how many writing rules to inject
    system_prompt     — pick a key from SYSTEM_PROMPTS
    user_template     — pick a key from USER_TEMPLATES (controls the
                        "task section" structure + where {query} sits)
    retrieval_mode    — "dense" | "bm25" | "hybrid"
    use_reranker      — True | False

Supported but limited by what's been pre-built:
    wg_chunker        — "a10" (with BM25 + tree expansion) or "c11"
                        (dense-only — set retrieval_mode="dense" with c11)
    articles_chunker  — only "a9" is currently embedded. To use another
                        chunker, run scripts.embed first.

Not supported (would require rebuilding indexes — out of scope here):
    embedder          — only e4 has been built. Other embedders would need
                        a full scripts.embed run first. Stick with e4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .retrieve import (
    bm25_search,
    expand_to_parents,
    expand_tree_to_leaves,
    rerank,
    rrf_fuse,
    vector_search,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "experiments"


# ===========================================================================
#  EDIT BELOW — all the knobs for your experiments
# ===========================================================================

BASE_QUERY = (
    "I want to write about how generative AI is changing advertising "
    "for Dutch businesses"
)

# ---- System prompts: pick by key in an experiment ------------------------

DEFAULT_SYSTEM_PROMPT = """\
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

TERSE_SYSTEM_PROMPT = (
    "You are an ABN AMRO business banking writer. Apply the supplied WRITING "
    "RULES strictly and match the supplied STYLE REFERENCES. Write in British "
    "English, 600-800 words, with ## subheadings. No invented stats, dates, "
    "names, quotes."
)

SYSTEM_PROMPTS: dict[str, str] = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "terse": TERSE_SYSTEM_PROMPT,
}


# ---- User templates: control the "task section" position + framing -------
# Placeholders: {rules}, {references}, {query}

DEFAULT_USER_TEMPLATE = (
    "WRITING RULES (apply strictly):\n\n{rules}\n\n"
    "---\n\n"
    "STYLE REFERENCES (match their tone/structure):\n\n{references}\n\n"
    "---\n\n"
    "TASK:\n\n{query}\n\n"
    "Write the article now."
)

TASK_FIRST_TEMPLATE = (
    "TASK:\n\n{query}\n\n"
    "---\n\n"
    "Apply these WRITING RULES strictly:\n\n{rules}\n\n"
    "---\n\n"
    "Use these STYLE REFERENCES to set tone and structure:\n\n{references}\n\n"
    "Write the article now."
)

# An aggressive rewording of the user's task — same query, richer framing.
DETAILED_TASK_TEMPLATE = (
    "WRITING RULES (apply strictly):\n\n{rules}\n\n"
    "---\n\n"
    "STYLE REFERENCES (match their tone/structure):\n\n{references}\n\n"
    "---\n\n"
    "TASK:\n\n"
    "Write an ABN AMRO Insights article in response to this brief: {query}\n\n"
    "Aim the piece at Dutch SME owners and finance leads. Lead with a concrete "
    "scenario, ground claims in the style references above, and close with a "
    "practical takeaway. Don't invent facts.\n\n"
    "Write the article now."
)

USER_TEMPLATES: dict[str, str] = {
    "default": DEFAULT_USER_TEMPLATE,
    "task_first": TASK_FIRST_TEMPLATE,
    "detailed_task": DETAILED_TASK_TEMPLATE,
}


# ---- The experiments themselves ------------------------------------------
# Each entry overrides any subset of DEFAULTS below. Anything omitted
# falls back to DEFAULTS. To add a new experiment, copy a block, rename,
# tweak a field. Keep `name` unique — it becomes the output folder.

EXPERIMENTS: list[dict] = [
    {
        # The current generate.py recipe, untouched. Use as the baseline
        # to diff every other experiment against.
        "name": "baseline",
        "query": BASE_QUERY,
    },
    # --- top-k swaps ------------------------------------------------------
    {
        "name": "more_rules",
        "query": BASE_QUERY,
        "top_k_rules": 10,
    },
    {
        "name": "more_articles",
        "query": BASE_QUERY,
        "top_k_articles": 6,
    },
    # --- retrieval mode swaps --------------------------------------------
    {
        "name": "dense_only",
        "query": BASE_QUERY,
        "retrieval_mode": "dense",
    },
    {
        "name": "bm25_only",
        "query": BASE_QUERY,
        "retrieval_mode": "bm25",
    },
    # --- reranker on/off --------------------------------------------------
    {
        "name": "no_rerank",
        "query": BASE_QUERY,
        "use_reranker": False,
    },
    # --- prompt swaps -----------------------------------------------------
    {
        "name": "terse_system",
        "query": BASE_QUERY,
        "system_prompt": "terse",
    },
    {
        "name": "task_first",
        "query": BASE_QUERY,
        "user_template": "task_first",
    },
    {
        "name": "detailed_task",
        "query": BASE_QUERY,
        "user_template": "detailed_task",
    },
    # --- chunker swap (writing-guide only — see note at top of file) -----
    {
        "name": "wg_chunker_c11",
        "query": BASE_QUERY,
        "wg_chunker": "c11",
        "retrieval_mode": "dense",  # c11 has no BM25 index built
    },
]


# ===========================================================================
#  Defaults — usually no need to edit these; override per-experiment instead
# ===========================================================================

DEFAULTS: dict = dict(
    # retrieval
    embedder="e4",
    articles_chunker="a9",
    articles_src="gpt-5",
    wg_chunker="a10",
    wg_src="writing-guide",
    top_n=30,
    top_k_articles=3,
    top_k_rules=5,
    retrieval_mode="hybrid",                     # "dense" | "bm25" | "hybrid"
    use_reranker=True,
    reranker_model="BAAI/bge-reranker-v2-m3",
    expand_parents=True,                          # a5/a9 small-to-big
    expand_tree=True,                             # a10 RAPTOR
    # prompts (keys into SYSTEM_PROMPTS / USER_TEMPLATES)
    system_prompt="default",
    user_template="default",
    # generation (OpenAI only)
    llm_model=None,                               # None => DEFAULT_OPENAI_MODEL
    max_tokens=2048,
    dry_run=False,
)

DEFAULT_OPENAI_MODEL = "gpt-4o"


# ===========================================================================
#  Pipeline — self-contained: retrieval, prompt composition, LLM call
# ===========================================================================


def _hybrid_retrieve(
    query: str,
    *,
    embedder: str,
    chunker: str,
    src: str,
    top_n: int,
    top_k: int,
    retrieval_mode: str,
    use_reranker: bool,
    reranker_model: str,
    expand_tree: bool,
    expand_parents_flag: bool,
) -> list[dict]:
    """Run the chosen retrieval pipeline for one corpus.

    retrieval_mode:
      "dense"  — embedder only
      "bm25"   — BM25 only
      "hybrid" — both, fused with RRF
    """
    if retrieval_mode == "dense":
        candidates = vector_search(query, embedder, chunker, src, top_n=top_n)
    elif retrieval_mode == "bm25":
        candidates = bm25_search(query, chunker, src, top_n=top_n)
    elif retrieval_mode == "hybrid":
        dense = vector_search(query, embedder, chunker, src, top_n=top_n)
        bm = bm25_search(query, chunker, src, top_n=top_n)
        candidates = rrf_fuse(dense, bm, k=60, top_n=top_n)
    else:
        raise SystemExit(f"unknown retrieval_mode: {retrieval_mode!r}")

    if expand_tree:
        candidates = expand_tree_to_leaves(candidates, chunker, src)

    if use_reranker and candidates:
        ranked = rerank(query, candidates, top_k=top_k, model_name=reranker_model)
    else:
        ranked = candidates[:top_k]

    if expand_parents_flag:
        ranked = expand_to_parents(ranked, dedup=True)

    return ranked


def _retrieve_style_references(query: str, cfg: dict) -> list[dict]:
    return _hybrid_retrieve(
        query,
        embedder=cfg["embedder"],
        chunker=cfg["articles_chunker"],
        src=cfg["articles_src"],
        top_n=cfg["top_n"],
        top_k=cfg["top_k_articles"],
        retrieval_mode=cfg["retrieval_mode"],
        use_reranker=cfg["use_reranker"],
        reranker_model=cfg["reranker_model"],
        expand_tree=False,                  # articles aren't a tree
        expand_parents_flag=cfg["expand_parents"],
    )


def _retrieve_writing_rules(query: str, cfg: dict) -> list[dict]:
    return _hybrid_retrieve(
        query,
        embedder=cfg["embedder"],
        chunker=cfg["wg_chunker"],
        src=cfg["wg_src"],
        top_n=cfg["top_n"],
        top_k=cfg["top_k_rules"],
        retrieval_mode=cfg["retrieval_mode"],
        use_reranker=cfg["use_reranker"],
        reranker_model=cfg["reranker_model"],
        expand_tree=cfg["expand_tree"],
        expand_parents_flag=False,          # parents not used for WG
    )


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


def _compose_user_message(template: str, query: str,
                          rules: list[dict], refs: list[dict]) -> str:
    rules_text = ("\n\n".join(_format_rule_block(r) for r in rules)
                  or "(no rules retrieved)")
    refs_text = ("\n\n".join(_format_reference_block(a) for a in refs)
                 or "(no references retrieved)")
    return template.format(rules=rules_text, references=refs_text, query=query)


def _call_openai(system: str, user: str, model: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ===========================================================================
#  Config merging + persistence
# ===========================================================================


def _merge(exp: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(exp)
    if not cfg.get("name"):
        raise SystemExit("each experiment must have a 'name'")
    if not cfg.get("query"):
        raise SystemExit(f"experiment {cfg['name']!r} missing 'query'")

    # resolve prompt-key aliases (so config.json shows the chosen key
    # AND the rendered text)
    if cfg["system_prompt"] not in SYSTEM_PROMPTS:
        raise SystemExit(
            f"experiment {cfg['name']!r}: unknown system_prompt "
            f"{cfg['system_prompt']!r}. Available: "
            f"{sorted(SYSTEM_PROMPTS)}"
        )
    if cfg["user_template"] not in USER_TEMPLATES:
        raise SystemExit(
            f"experiment {cfg['name']!r}: unknown user_template "
            f"{cfg['user_template']!r}. Available: "
            f"{sorted(USER_TEMPLATES)}"
        )
    if cfg["retrieval_mode"] not in ("dense", "bm25", "hybrid"):
        raise SystemExit(
            f"experiment {cfg['name']!r}: unknown retrieval_mode "
            f"{cfg['retrieval_mode']!r}"
        )
    if cfg["llm_model"] is None:
        cfg["llm_model"] = DEFAULT_OPENAI_MODEL
    return cfg


def _serialize_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "id": h.get("id"),
            "text": h.get("text"),
            "metadata": h.get("metadata"),
            "distance": h.get("distance"),
            "rrf_score": h.get("rrf_score"),
            "rerank_score": h.get("rerank_score"),
            "bm25_score": h.get("bm25_score"),
        }
        for h in hits
    ]


def run_experiment(cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]

    print(f"[experiment:{name}] retrieving style references "
          f"(mode={cfg['retrieval_mode']}, rerank={cfg['use_reranker']})...",
          file=sys.stderr)
    t0 = time.time()
    references = _retrieve_style_references(cfg["query"], cfg)
    print(f"[experiment:{name}] retrieving writing rules...", file=sys.stderr)
    rules = _retrieve_writing_rules(cfg["query"], cfg)
    t_retrieval = time.time() - t0

    system_prompt = SYSTEM_PROMPTS[cfg["system_prompt"]]
    user_template = USER_TEMPLATES[cfg["user_template"]]
    user_message = _compose_user_message(user_template, cfg["query"],
                                         rules, references)

    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False))
    (out_dir / "retrieved_references.json").write_text(
        json.dumps(_serialize_hits(references), indent=2, ensure_ascii=False))
    (out_dir / "retrieved_rules.json").write_text(
        json.dumps(_serialize_hits(rules), indent=2, ensure_ascii=False))
    (out_dir / "system_prompt.txt").write_text(system_prompt)
    (out_dir / "user_message.txt").write_text(user_message)

    output = None
    t_llm = None
    if cfg["dry_run"]:
        print(f"[experiment:{name}] dry-run — skipping LLM call",
              file=sys.stderr)
    else:
        model = cfg["llm_model"]
        print(f"[experiment:{name}] calling openai ({model})...",
              file=sys.stderr)
        t1 = time.time()
        output = _call_openai(system_prompt, user_message,
                              model, cfg["max_tokens"])
        t_llm = time.time() - t1
        (out_dir / "output.md").write_text(output)

    summary = {
        "name": name,
        "query": cfg["query"],
        "retrieval_mode": cfg["retrieval_mode"],
        "use_reranker": cfg["use_reranker"],
        "system_prompt_key": cfg["system_prompt"],
        "user_template_key": cfg["user_template"],
        "n_references": len(references),
        "n_rules": len(rules),
        "system_prompt_chars": len(system_prompt),
        "user_message_chars": len(user_message),
        "total_prompt_chars": len(system_prompt) + len(user_message),
        "approx_prompt_tokens": (len(system_prompt) + len(user_message)) // 4,
        "output_chars": len(output) if output else 0,
        "approx_output_tokens": (len(output) // 4) if output else 0,
        "retrieval_seconds": round(t_retrieval, 2),
        "llm_seconds": round(t_llm, 2) if t_llm is not None else None,
        "llm_model": cfg["llm_model"],
        "embedder": cfg["embedder"],
        "articles_chunker": cfg["articles_chunker"],
        "wg_chunker": cfg["wg_chunker"],
        "top_k_articles": cfg["top_k_articles"],
        "top_k_rules": cfg["top_k_rules"],
        "dry_run": cfg["dry_run"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ===========================================================================
#  Compare past runs
# ===========================================================================


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])


def _resolve_run(spec: str) -> Path:
    if spec == "latest":
        runs = _list_runs()
        if not runs:
            raise SystemExit("no past runs to compare")
        return runs[-1]
    p = RUNS_DIR / spec
    if p.is_dir():
        return p
    raise SystemExit(f"run not found: {spec}")


def compare_run(run_dir: Path) -> None:
    exps = sorted([d for d in run_dir.iterdir() if d.is_dir()])
    rows = []
    for d in exps:
        sp = d / "summary.json"
        if sp.exists():
            rows.append(json.loads(sp.read_text()))
    if not rows:
        print(f"(no summaries in {run_dir})")
        return

    print(f"\nRun: {run_dir.name}    ({len(rows)} experiments)\n")
    cols = [
        ("name", 18),
        ("mode", 7),
        ("rerank", 6),
        ("n_refs", 6),
        ("n_rules", 7),
        ("prompt_tok", 10),
        ("out_tok", 7),
        ("retr_s", 6),
        ("llm_s", 6),
        ("sys", 8),
        ("tmpl", 13),
    ]
    header = " | ".join(f"{c:<{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = {
            "name": r["name"][:18],
            "mode": r["retrieval_mode"],
            "rerank": "yes" if r["use_reranker"] else "no",
            "n_refs": r["n_references"],
            "n_rules": r["n_rules"],
            "prompt_tok": r["approx_prompt_tokens"],
            "out_tok": r["approx_output_tokens"],
            "retr_s": r["retrieval_seconds"],
            "llm_s": r["llm_seconds"] if r["llm_seconds"] is not None else "-",
            "sys": r["system_prompt_key"][:8],
            "tmpl": r["user_template_key"][:13],
        }
        print(" | ".join(f"{str(vals[c]):<{w}}" for c, w in cols))

    print()
    print("Output previews (first 240 chars):")
    print("-" * 78)
    for d in exps:
        out_path = d / "output.md"
        if not out_path.exists():
            print(f"\n[{d.name}] (no output — dry-run?)")
            continue
        snippet = out_path.read_text().strip().replace("\n", " ")
        print(f"\n[{d.name}] {snippet[:240]}{'...' if len(snippet) > 240 else ''}")

    print()
    print(f"Artifacts per experiment (under {run_dir}/<name>/):")
    print("  config.json  retrieved_references.json  retrieved_rules.json")
    print("  system_prompt.txt  user_message.txt  output.md  summary.json")


# ===========================================================================
#  CLI
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config",
                    help="JSON file with a [{...}, ...] list of experiments; "
                         "defaults to EXPERIMENTS in this file")
    ap.add_argument("--only",
                    help="comma-separated experiment names to run (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip LLM calls for every experiment")
    ap.add_argument("--run-id", help="custom run dir name (default: timestamp)")
    ap.add_argument("--list", action="store_true",
                    help="list defined experiments and exit")
    ap.add_argument("--list-runs", action="store_true",
                    help="list past run-ids and exit")
    ap.add_argument("--compare",
                    help="print a comparison table for a past run "
                         "(use 'latest' or a run-id)")
    args = ap.parse_args()

    if args.compare:
        compare_run(_resolve_run(args.compare))
        return
    if args.list_runs:
        for r in _list_runs():
            print(r.name)
        return

    if args.config:
        raw = json.loads(Path(args.config).read_text())
    else:
        raw = EXPERIMENTS

    if args.list:
        for e in raw:
            print(f"  {e.get('name', '?')}: {(e.get('query') or '?')[:80]}")
        return

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    selected = [e for e in raw if only is None or e.get("name") in only]
    if not selected:
        raise SystemExit(f"no experiments matched --only={args.only!r}")

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for e in selected:
        cfg = _merge(e)
        if args.dry_run:
            cfg["dry_run"] = True
        out_dir = run_dir / cfg["name"]
        summaries.append(run_experiment(cfg, out_dir))

    (run_dir / "_summaries.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\n[experiment] wrote {len(summaries)} experiment(s) to {run_dir}")
    print(f"[experiment] compare with: "
          f"python -m scripts.experiment --compare {run_id}")


if __name__ == "__main__":
    main()
