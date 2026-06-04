"""Retrieval — Chroma + BM25 + RRF fusion + cross-encoder rerank.

Pipeline:
  1. Candidate generation:
       --mode dense  → vector search only (default until v0.2)
       --mode bm25   → BM25 keyword search only
       --mode hybrid → both, fused with Reciprocal Rank Fusion (RRF, k=60)
  2. (Optional) Cross-encoder reranker scores (query, candidate) pairs (X1).
  3. For small-to-big chunkers (a5, a9), expand each hit's text to the
     parent_text stored in metadata.

Usage:
    # Default — hybrid retrieval + reranker + parent expansion
    python -m scripts.retrieve --query "5G policy delay"

    # Compare dense-only vs hybrid on the same query
    python -m scripts.retrieve --query "5G policy delay" --mode dense
    python -m scripts.retrieve --query "5G policy delay" --mode hybrid

    # BM25 only (no embedder needed)
    python -m scripts.retrieve --query "NIS2 17 October 2024" --mode bm25

    # Dense vs hybrid before/after (visible diff)
    python -m scripts.retrieve --query "NIS2 deadline" --compare-modes
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_BASE = ROOT / "vector_db"

# ---------------------------------------------------------------------------
# Embedder query-side prefixes (mirrors embed.py EMBEDDERS table)
# ---------------------------------------------------------------------------

QUERY_PREFIXES = {
    "e1": "query: ",
    "e1i": (
        "Instruct: Given a search query about ABN AMRO insight articles, "
        "retrieve relevant passages that answer the query\nQuery: "
    ),
    "e2": "query: ",
    "e3": "",
    "e4": "",
    "e5": "Represent this sentence for searching relevant passages: ",
    "e6": "",
}

EMBEDDER_DIRS = {
    "e1": "e1-multilingual-e5-large",
    "e1i": "e1i-multilingual-e5-large-instruct",
    "e2": "e2-multilingual-e5-base",
    "e3": "e3-all-MiniLM-L6-v2",
    "e4": "e4-bge-m3",
    "e5": "e5-bge-large-en-v1.5",
    "e6": "e6-jina-embeddings-v3",
}

EMBEDDER_MODELS = {
    "e1": "intfloat/multilingual-e5-large",
    "e1i": "intfloat/multilingual-e5-large-instruct",
    "e2": "intfloat/multilingual-e5-base",
    "e3": "sentence-transformers/all-MiniLM-L6-v2",
    "e4": "BAAI/bge-m3",
    "e5": "BAAI/bge-large-en-v1.5",
    "e6": "jinaai/jina-embeddings-v3",
}


# ---------------------------------------------------------------------------
# Embedding the query
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[str, object] = {}


def _load_st_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        print(f"[retrieve] loading embedder {model_name} ...", file=sys.stderr)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _MODEL_CACHE[model_name]


def embed_query(query: str, embedder: str) -> list[float]:
    if embedder not in EMBEDDER_MODELS:
        raise SystemExit(f"unknown embedder for retrieval: {embedder!r}")
    model = _load_st_model(EMBEDDER_MODELS[embedder])
    prefix = QUERY_PREFIXES.get(embedder, "")
    vec = model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)
    return vec[0].tolist()


# ---------------------------------------------------------------------------
# Resolving collection name from chunker shorthand
# ---------------------------------------------------------------------------


def _find_chunker_folder(chunker: str) -> str:
    """Match 'c1' -> 'c1-fixed-512-50', accept full folder names too."""
    chunked_base = ROOT / "chunked"
    for tier in ("simple", "advanced"):
        d = chunked_base / tier
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_dir() and (f.name == chunker or f.name.split("-")[0] == chunker):
                return f.name
    raise SystemExit(f"chunker {chunker!r} not found under data/chunked/")


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


def vector_search(
    query: str,
    embedder: str,
    chunker: str,
    src: str,
    top_n: int = 50,
    where: dict | None = None,
) -> list[dict]:
    """Returns list of {id, text, metadata, distance}."""
    import chromadb

    db_path = VECTOR_DB_BASE / EMBEDDER_DIRS[embedder]
    if not db_path.exists():
        raise SystemExit(f"no embedder DB at {db_path} — run embed.py first")
    client = chromadb.PersistentClient(path=str(db_path))
    coll_name = f"{_find_chunker_folder(chunker)}__{src}"
    coll = client.get_collection(coll_name)

    qvec = embed_query(query, embedder)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=min(top_n, coll.count()),
        where=where,
    )
    return [
        {
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]


# ---------------------------------------------------------------------------
# BM25 search (X4)
# ---------------------------------------------------------------------------

_BM25_TOKEN_RE = re.compile(r"\w+")


def _bm25_tokenize(text: str) -> list[str]:
    """Mirror of BM25Backend.tokenize in embed.py — lowercase + word-split."""
    return _BM25_TOKEN_RE.findall(text.lower())


_BM25_CACHE: dict[Path, tuple[object, list[dict]]] = {}


def _load_bm25(chunker_folder: str, src: str) -> tuple[object, list[dict]]:
    """Load the BM25 index and docs for a (chunker, src) pair, cached."""
    key = VECTOR_DB_BASE / "x4-bm25" / f"{chunker_folder}__{src}"
    if not key.exists():
        raise SystemExit(
            f"BM25 index not found at {key}\n"
            f"Run: python -m scripts.embed --embedder x4 --chunker {chunker_folder} --src {src}"
        )
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]
    print(f"[retrieve] loading BM25 index for {key.name} ...", file=sys.stderr)
    with (key / "index.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    docs = [json.loads(line) for line in (key / "docs.jsonl").open(encoding="utf-8")]
    _BM25_CACHE[key] = (bm25, docs)
    return bm25, docs


def bm25_search(
    query: str, chunker: str, src: str, top_n: int = 50,
) -> list[dict]:
    """BM25 keyword search. Returns same shape as vector_search:
    list of {id, text, metadata, bm25_score}."""
    chunker_folder = _find_chunker_folder(chunker)
    bm25, docs = _load_bm25(chunker_folder, src)
    tokens = _bm25_tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_n]
    return [
        {
            "id": docs[i]["id"],
            "text": docs[i]["text"],
            "metadata": docs[i]["metadata"],
            "bm25_score": float(score),
        }
        for i, score in ranked
        if score > 0  # drop zero-score hits (no token match)
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (hybrid retrieval)
# ---------------------------------------------------------------------------


def rrf_fuse(
    *result_lists: list[dict], k: int = 60, top_n: int = 50,
) -> list[dict]:
    """Reciprocal Rank Fusion. Combines ranked lists into one score-ordered
    list. Documents present in multiple lists get summed contributions —
    consensus picks float to the top.

    RRF_score(d) = sum over each list:  1 / (k + rank_in_that_list + 1)

    The standard k=60 dampens the influence of any single list's top hit so
    a doc found by multiple retrievers beats one only found by the strongest.
    """
    score_by_id: dict[str, float] = {}
    doc_by_id: dict[str, dict] = {}
    rank_sources: dict[str, list[tuple[int, int]]] = {}  # id -> [(list_idx, rank), ...]

    for list_idx, results in enumerate(result_lists):
        for rank, hit in enumerate(results):
            doc_id = hit["id"]
            score_by_id[doc_id] = score_by_id.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in doc_by_id:
                doc_by_id[doc_id] = hit
            rank_sources.setdefault(doc_id, []).append((list_idx, rank + 1))

    fused = sorted(score_by_id.items(), key=lambda x: -x[1])[:top_n]
    out: list[dict] = []
    for doc_id, rrf_score in fused:
        merged = dict(doc_by_id[doc_id])
        merged["rrf_score"] = rrf_score
        merged["rrf_sources"] = rank_sources[doc_id]  # for debugging
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# Cross-encoder reranker (X1)
# ---------------------------------------------------------------------------


_RERANKER_CACHE: dict[str, object] = {}

# Default: BGE multilingual reranker — handles NL+EN, ~600 MB, no API key.
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def _load_reranker(model_name: str):
    if model_name not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder

        print(f"[retrieve] loading reranker {model_name} ...", file=sys.stderr)
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name, trust_remote_code=True)
    return _RERANKER_CACHE[model_name]


def rerank(
    query: str, candidates: list[dict], top_k: int = 5,
    model_name: str = DEFAULT_RERANKER,
) -> list[dict]:
    """Score (query, doc) pairs with a cross-encoder; return top_k by score."""
    if not candidates:
        return []
    model = _load_reranker(model_name)
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    out = []
    for c, s in zip(candidates, scores):
        c2 = dict(c)
        c2["rerank_score"] = float(s)
        out.append(c2)
    out.sort(key=lambda x: -x["rerank_score"])
    return out[:top_k]


# ---------------------------------------------------------------------------
# Small-to-big parent expansion
# ---------------------------------------------------------------------------


def expand_to_parents(hits: list[dict], dedup: bool = True) -> list[dict]:
    """For chunkers that store parent_text in metadata (a5, a9), swap each
    hit's text for the parent. Optionally deduplicate by parent_index so the
    same parent isn't returned twice."""
    seen_parents: set[tuple] = set()
    out: list[dict] = []
    for h in hits:
        m = h["metadata"]
        if "parent_text" not in m:
            out.append(h)
            continue
        slug = m.get("source_slug")
        pi = m.get("parent_index")
        key = (slug, pi)
        if dedup and key in seen_parents:
            continue
        seen_parents.add(key)
        h2 = dict(h)
        h2["text"] = m["parent_text"]
        h2["_was_expanded"] = True
        out.append(h2)
    return out


# ---------------------------------------------------------------------------
# A10 RAPTOR-structural: Pattern A expansion
#   Match at any tree level → expand non-leaf hits to their descendant leaves
#   → rerank → send only leaves to LLM
# ---------------------------------------------------------------------------


def expand_tree_to_leaves(
    hits: list[dict], chunker: str, src: str, dedup: bool = True,
) -> list[dict]:
    """For RAPTOR-style chunkers (a10) that store `descendant_leaf_ids` and
    `depth` in metadata: replace each non-leaf hit with its descendant leaves.
    Leaf hits pass through. Non-leaves with no descendants (sections that
    didn't split into rules) keep their own content as a fallback leaf."""
    chunker_folder = _find_chunker_folder(chunker)
    # Lazily build an in-memory map of node_id -> chunk record from the JSONL
    chunked_path = ROOT / "chunked"
    candidate_paths = [
        chunked_path / "advanced" / chunker_folder / src / "writing_guide.jsonl",
        chunked_path / "simple"   / chunker_folder / src / "writing_guide.jsonl",
    ]
    src_jsonl = next((p for p in candidate_paths if p.exists()), None)
    if src_jsonl is None:
        # nothing to expand against — fall back to original hits
        return hits

    nodes_by_id: dict[str, dict] = {}
    with src_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # `descendant_leaf_ids` stores node_ids — index by node_id, not chunk_id
            node_id = row.get("metadata", {}).get("node_id") or row["id"]
            nodes_by_id[node_id] = row

    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        m = h["metadata"]
        depth = m.get("depth")
        if depth == 3:  # leaf already
            if dedup and h["id"] in seen:
                continue
            seen.add(h["id"])
            out.append(h)
            continue
        # non-leaf — try to expand
        desc_ids_str = m.get("descendant_leaf_ids", "") or ""
        desc_ids = [d for d in desc_ids_str.split(",") if d]
        if not desc_ids:
            # leaf-less section: fall back to its own content
            if dedup and h["id"] in seen:
                continue
            seen.add(h["id"])
            h2 = dict(h)
            h2["_expanded_from_non_leaf"] = "self_fallback"
            out.append(h2)
            continue
        for did in desc_ids:
            if dedup and did in seen:
                continue
            row = nodes_by_id.get(did)
            if not row:
                continue
            seen.add(did)
            out.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": row["metadata"],
                "distance": h.get("distance", 0.0),
                "_expanded_from_non_leaf": h["id"],
            })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_hit(i: int, h: dict, max_text: int = 200) -> str:
    m = h["metadata"]
    title = m.get("source_title", "?")
    if "rerank_score" in h:
        score = f"rerank={h['rerank_score']:.3f}"
    elif "rrf_score" in h:
        sources = h.get("rrf_sources", [])
        src_tag = "/".join(["dense" if s == 0 else "bm25" for s, _ in sources])
        score = f"rrf={h['rrf_score']:.4f} ({src_tag})"
    elif "bm25_score" in h:
        score = f"bm25={h['bm25_score']:.2f}"
    else:
        score = f"dist={h['distance']:.3f}"
    txt = h["text"].replace("\n", " ").strip()[:max_text]
    return f"  {i+1}. [{score}] {title!r}\n     {txt!r}..."


def _retrieve_candidates(query: str, mode: str, args) -> list[dict]:
    """Run the chosen candidate-generation strategy. Returns top_n results."""
    if mode == "dense":
        return vector_search(
            query, args.embedder, args.chunker, args.src, top_n=args.top_n,
        )
    if mode == "bm25":
        return bm25_search(query, args.chunker, args.src, top_n=args.top_n)
    if mode == "hybrid":
        dense = vector_search(
            query, args.embedder, args.chunker, args.src, top_n=args.top_n,
        )
        bm = bm25_search(query, args.chunker, args.src, top_n=args.top_n)
        return rrf_fuse(dense, bm, k=args.rrf_k, top_n=args.top_n)
    raise SystemExit(f"unknown --mode: {mode!r}")


def _run_pipeline(query: str, mode: str, args) -> tuple[list[dict], list[dict]]:
    """Returns (candidates_before_rerank, final_ranked).

    For RAPTOR-tree chunkers (a10), candidates first get expanded from
    non-leaf nodes to their descendant leaves BEFORE the reranker runs —
    so the reranker scores leaves only, and the final output is always
    leaf-level rules (Pattern A: match anywhere, send leaves)."""
    candidates = _retrieve_candidates(query, mode, args)
    # Pattern A: expand RAPTOR non-leaf hits into descendant leaves
    if args.expand_tree:
        candidates = expand_tree_to_leaves(candidates, args.chunker, args.src)
    if args.rerank and candidates:
        ranked = rerank(
            query, candidates, top_k=args.top_k, model_name=args.reranker_model,
        )
    else:
        ranked = candidates[: args.top_k]
    if args.expand_parents:
        ranked = expand_to_parents(ranked, dedup=True)
    return candidates, ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--mode", default="hybrid", choices=("dense", "bm25", "hybrid"),
                    help="candidate-generation strategy")
    ap.add_argument("--embedder", default="e4", choices=sorted(EMBEDDER_MODELS),
                    help="dense embedder (ignored for --mode bm25)")
    ap.add_argument("--chunker", default="a9",
                    help="chunker short id (c1, a9, ...) or full folder name")
    ap.add_argument("--src", default="gpt-5",
                    help="source method: gpt-5, deep-translator, source-nl, writing-guide")
    ap.add_argument("--top-n", type=int, default=50,
                    help="candidates to generate (per retriever) before reranking")
    ap.add_argument("--top-k", type=int, default=5,
                    help="final results to return")
    ap.add_argument("--rrf-k", type=int, default=60,
                    help="RRF damping constant (only used for --mode hybrid)")
    ap.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True,
                    help="apply cross-encoder reranker (X1)")
    ap.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    ap.add_argument("--expand-parents", action=argparse.BooleanOptionalAction, default=True,
                    help="for a5/a9, return parent_text instead of child_text")
    ap.add_argument("--expand-tree", action=argparse.BooleanOptionalAction, default=True,
                    help="for a10 RAPTOR-tree, expand non-leaf hits to their descendant leaves "
                         "before reranking (Pattern A)")
    ap.add_argument("--compare", action="store_true",
                    help="show before/after rerank for the chosen mode")
    ap.add_argument("--compare-modes", action="store_true",
                    help="run dense, bm25, and hybrid side-by-side")
    args = ap.parse_args()

    print(f"\n[query] {args.query!r}", file=sys.stderr)
    print(
        f"[config] mode={args.mode} embedder={args.embedder} chunker={args.chunker} "
        f"src={args.src} top_n={args.top_n} top_k={args.top_k} "
        f"rerank={args.rerank} expand_parents={args.expand_parents}",
        file=sys.stderr,
    )

    if args.compare_modes:
        for mode in ("dense", "bm25", "hybrid"):
            print(f"\n=== mode={mode} ===")
            try:
                _, ranked = _run_pipeline(args.query, mode, args)
                for i, h in enumerate(ranked):
                    print(_format_hit(i, h))
            except SystemExit as e:
                print(f"  (skipped: {e})")
        return

    candidates, ranked = _run_pipeline(args.query, args.mode, args)

    if args.compare and args.rerank:
        print(f"\n=== BEFORE rerank (mode={args.mode}, top-K) ===")
        for i, h in enumerate(candidates[: args.top_k]):
            print(_format_hit(i, h))
        print(f"\n=== AFTER rerank (top-K) ===")
        for i, h in enumerate(ranked):
            print(_format_hit(i, h))
    else:
        print("\n=== Top results ===")
        for i, h in enumerate(ranked):
            print(_format_hit(i, h))


if __name__ == "__main__":
    main()
