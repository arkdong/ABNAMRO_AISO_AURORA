"""Embed chunked outputs into Chroma collections.

Reads JSONL from `data/chunked/<tier>/<chunker>/<src>/*.jsonl` and produces
per-embedder Chroma databases at `vector_db/<embedder_id>/`. Each unique
(chunker × src) gets its own collection inside the embedder's DB.

Why one DB per embedder, not one DB total: different embedders produce
different-dim vectors and live in different vector spaces. Mixing them in
the same collection breaks retrieval. Per-embedder DBs make swapping
embedders a directory-level operation.

Embedder registry (E1-E9 from docs/experiments.md):
  e1  intfloat/multilingual-e5-large       multilingual, 1024-dim, e5-prefix
  e2  intfloat/multilingual-e5-base        multilingual, 768-dim,  e5-prefix
  e3  sentence-transformers/all-MiniLM-L6-v2  English-only, 384-dim, no prefix
  e4  BAAI/bge-m3                          multilingual, 1024-dim, no prefix
  e5  BAAI/bge-large-en-v1.5               English, 1024-dim, BGE-prefix
  e6  jinaai/jina-embeddings-v3            multilingual, 1024-dim, task-prefix
  e7  openai/text-embedding-3-large        API, 3072-dim
  e8  openai/text-embedding-3-small        API, 1536-dim
  e9  cohere/embed-multilingual-v3.0       API, 1024-dim

IMPORTANT: most multilingual sentence-transformer models REQUIRE a prefix
("passage: ..." for indexed chunks, "query: ..." at retrieval time). The
backends below handle this automatically. Skipping the prefix silently
drops precision by 5-15% — see docs/advanced-embedding.md X5.

Usage:
    python -m scripts.embed --embedder e2 --chunker c1 --src deep-translator
    python -m scripts.embed --embedder e3 --chunker all-no-key
    python -m scripts.embed --embedder all-no-key --chunker c1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CHUNKED_BASE = ROOT / "data" / "chunked"
VECTOR_DB_BASE = ROOT / "vector_db"


# ---------------------------------------------------------------------------
# Embedder registry
# ---------------------------------------------------------------------------

EMBEDDERS: dict[str, dict] = {
    "e1": {
        "kind": "sentence-transformers",
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "needs_key": False,
    },
    "e2": {
        "kind": "sentence-transformers",
        "model": "intfloat/multilingual-e5-base",
        "dim": 768,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "needs_key": False,
    },
    "e3": {
        "kind": "sentence-transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "passage_prefix": "",
        "query_prefix": "",
        "needs_key": False,
    },
    "e4": {
        "kind": "sentence-transformers",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "passage_prefix": "",
        "query_prefix": "",
        "needs_key": False,
    },
    "e5": {
        "kind": "sentence-transformers",
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "passage_prefix": "",
        "query_prefix": (
            # BGE recommends prepending an instruction at query time only.
            "Represent this sentence for searching relevant passages: "
        ),
        "needs_key": False,
    },
    "e6": {
        "kind": "sentence-transformers",
        "model": "jinaai/jina-embeddings-v3",
        "dim": 1024,
        "passage_prefix": "",
        "query_prefix": "",
        "needs_key": False,
    },
    "e7": {
        "kind": "openai",
        "model": "text-embedding-3-large",
        "dim": 3072,
        "needs_key": True,
        "env_var": "OPENAI_API_KEY",
    },
    "e8": {
        "kind": "openai",
        "model": "text-embedding-3-small",
        "dim": 1536,
        "needs_key": True,
        "env_var": "OPENAI_API_KEY",
    },
    "e9": {
        "kind": "cohere",
        "model": "embed-multilingual-v3.0",
        "dim": 1024,
        "needs_key": True,
        "env_var": "COHERE_API_KEY",
    },
}


def embedder_id(eid: str) -> str:
    """Folder-friendly id like 'e1-multilingual-e5-large'."""
    info = EMBEDDERS[eid]
    short = info["model"].split("/")[-1]
    return f"{eid}-{short}"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class STBackend:
    def __init__(self, info: dict):
        from sentence_transformers import SentenceTransformer

        print(f"[embed] loading {info['model']} ...", file=sys.stderr)
        # trust_remote_code needed for jina-v3 and similar
        self.model = SentenceTransformer(info["model"], trust_remote_code=True)
        self.passage_prefix = info.get("passage_prefix", "")
        self.query_prefix = info.get("query_prefix", "")
        self.dim = info["dim"]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.passage_prefix + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.tolist()


class OpenAIBackend:
    BATCH = 100  # OpenAI accepts up to 2048 inputs per call; keep it modest.

    def __init__(self, info: dict):
        if not os.getenv(info["env_var"]):
            raise SystemExit(f"{info['env_var']} not set — required for {info['model']}.")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = info["model"]
        self.dim = info["dim"]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


class CohereBackend:
    BATCH = 96  # Cohere v3 supports up to 96 texts per request.

    def __init__(self, info: dict):
        if not os.getenv(info["env_var"]):
            raise SystemExit(f"{info['env_var']} not set — required for {info['model']}.")
        try:
            import cohere
        except ImportError as e:
            raise SystemExit("Install cohere: pip install cohere") from e
        self.client = cohere.Client(os.environ[info["env_var"]])
        self.model = info["model"]
        self.dim = info["dim"]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            resp = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
            )
            out.extend(resp.embeddings)
        return out


def make_backend(eid: str):
    info = EMBEDDERS[eid]
    kind = info["kind"]
    if kind == "sentence-transformers":
        return STBackend(info)
    if kind == "openai":
        return OpenAIBackend(info)
    if kind == "cohere":
        return CohereBackend(info)
    raise SystemExit(f"unknown embedder kind: {kind!r}")


# ---------------------------------------------------------------------------
# Chunked I/O
# ---------------------------------------------------------------------------


def find_chunker_dir(chunker: str, src: str) -> Path | None:
    """Locate data/chunked/<tier>/<folder>/<src>/. Accepts chunker as full
    folder name (e.g. 'c1-fixed-512-50') or short id (e.g. 'c1')."""
    for tier in ("simple", "advanced"):
        tier_dir = CHUNKED_BASE / tier
        if not tier_dir.exists():
            continue
        for folder in tier_dir.iterdir():
            if not folder.is_dir():
                continue
            if folder.name == chunker or folder.name.split("-")[0] == chunker:
                p = folder / src
                if p.exists():
                    return p
    return None


def list_chunkers_with_data(src: str) -> list[tuple[str, Path]]:
    """Return [(chunker_folder_name, dir_path), ...] for every chunker that
    has output for the given src."""
    out: list[tuple[str, Path]] = []
    for tier in ("simple", "advanced"):
        tier_dir = CHUNKED_BASE / tier
        if not tier_dir.exists():
            continue
        for folder in sorted(tier_dir.iterdir()):
            if not folder.is_dir():
                continue
            d = folder / src
            if d.exists() and any(d.glob("*.jsonl")):
                out.append((folder.name, d))
    return out


def load_chunks_from_dir(d: Path) -> Iterable[dict]:
    for jsonl in sorted(d.glob("*.jsonl")):
        if jsonl.name.startswith("_"):
            continue
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool/None.
    Convert lists to comma-joined strings, drop nested dicts, drop None-valued
    keys to keep collections tidy."""
    out: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = " > ".join(str(x) for x in v) if v else ""
        elif isinstance(v, dict):
            # flatten one level: sub_a, sub_b => key.sub_a, key.sub_b
            for sk, sv in v.items():
                if isinstance(sv, (str, int, float, bool)):
                    out[f"{k}.{sk}"] = sv
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------


def collection_name(chunker_folder: str, src: str) -> str:
    return f"{chunker_folder}__{src}"


def embed_one(
    eid: str,
    backend,
    chunker_folder: str,
    chunker_dir: Path,
    src: str,
    force: bool,
    batch_size: int = 256,
) -> dict:
    import chromadb

    db_path = VECTOR_DB_BASE / embedder_id(eid)
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    coll_name = collection_name(chunker_folder, src)
    coll = client.get_or_create_collection(name=coll_name)

    # Skip-if-already-populated logic
    existing = coll.count()
    chunks = list(load_chunks_from_dir(chunker_dir))
    n_total = len(chunks)
    if existing >= n_total and not force:
        return {
            "embedder": eid,
            "chunker": chunker_folder,
            "src": src,
            "collection": coll_name,
            "chunks_in_jsonl": n_total,
            "chunks_in_chroma": existing,
            "skipped_existing": True,
            "elapsed_seconds": 0.0,
        }
    if force:
        # delete and recreate to ensure clean state
        client.delete_collection(coll_name)
        coll = client.get_or_create_collection(name=coll_name)

    t0 = time.time()
    n_added = 0
    for start in range(0, n_total, batch_size):
        batch = chunks[start : start + batch_size]
        ids = [c["id"] for c in batch]
        docs = [c["text"] for c in batch]
        metas = [sanitize_metadata(c["metadata"]) for c in batch]
        vecs = backend.encode_passages(docs)
        coll.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        n_added += len(batch)

    return {
        "embedder": eid,
        "chunker": chunker_folder,
        "src": src,
        "collection": coll_name,
        "chunks_in_jsonl": n_total,
        "chunks_added": n_added,
        "skipped_existing": False,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def expand_chunker_arg(arg: str, src: str) -> list[tuple[str, Path]]:
    """Resolve --chunker arg to one or more (folder_name, dir) tuples."""
    if arg in ("all-no-key", "all"):
        # all chunkers that have data for this src
        return list_chunkers_with_data(src)
    if arg.startswith("all-simple"):
        return [(n, d) for n, d in list_chunkers_with_data(src)
                if (CHUNKED_BASE / "simple" / n).exists()]
    if arg.startswith("all-advanced"):
        return [(n, d) for n, d in list_chunkers_with_data(src)
                if (CHUNKED_BASE / "advanced" / n).exists()]
    # single chunker
    d = find_chunker_dir(arg, src)
    if d is None:
        raise SystemExit(f"chunker {arg!r} has no data for src={src!r}")
    return [(d.parent.name, d)]


def expand_embedder_arg(arg: str) -> list[str]:
    if arg == "all":
        return list(EMBEDDERS.keys())
    if arg == "all-no-key":
        return [e for e, info in EMBEDDERS.items() if not info["needs_key"]]
    if arg in EMBEDDERS:
        return [arg]
    raise SystemExit(f"unknown embedder: {arg!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedder", required=True,
        help=f"e1..e9, 'all', or 'all-no-key' (E1-E6 local). Available: {sorted(EMBEDDERS)}",
    )
    ap.add_argument(
        "--chunker", default="all-no-key",
        help="chunker id (e.g. c1) or folder name (c1-fixed-512-50), "
             "or 'all'/'all-simple'/'all-advanced'/'all-no-key'",
    )
    ap.add_argument("--src", default="deep-translator",
                    help="translation method, or 'writing-guide' for WG-only chunkers")
    ap.add_argument("--force", action="store_true",
                    help="re-embed even if collection is already populated")
    args = ap.parse_args()

    # When --chunker matches both translated and writing-guide chunkers, the
    # caller picks src. Iterate explicitly over both sources so all-* "just works".
    src_options = [args.src]
    if args.chunker.startswith("all"):
        # also check writing-guide src for WG-only chunkers
        if args.src != "writing-guide":
            src_options.append("writing-guide")

    eids = expand_embedder_arg(args.embedder)
    print(f"[embed] embedders: {eids}", file=sys.stderr)

    summary: list[dict] = []
    for eid in eids:
        backend = make_backend(eid)  # load model once per embedder
        for src in src_options:
            targets = expand_chunker_arg(args.chunker, src)
            for chunker_folder, chunker_dir in targets:
                try:
                    info = embed_one(eid, backend, chunker_folder, chunker_dir, src, args.force)
                    print(
                        f"[embed] {eid} × {chunker_folder} × {src} → "
                        f"chunks={info.get('chunks_added', info.get('chunks_in_jsonl'))} "
                        f"elapsed={info.get('elapsed_seconds')}s "
                        f"{'(skipped, already populated)' if info.get('skipped_existing') else ''}",
                        file=sys.stderr,
                    )
                    summary.append(info)
                except Exception as e:
                    print(f"[embed] {eid} × {chunker_folder} × {src} → ERROR: {e}", file=sys.stderr)

    # Write summary
    summary_path = VECTOR_DB_BASE / "_last_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[embed] summary -> {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
