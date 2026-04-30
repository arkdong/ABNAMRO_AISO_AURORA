"""Embed chunked outputs into Chroma collections (E1-E9) or build advanced
indexes (X4 BM25, X7 BGE-M3 multi-output, X8 Matryoshka truncation).

Dense models read JSONL from `data/chunked/<tier>/<chunker>/<src>/*.jsonl`
and produce per-embedder Chroma databases at `vector_db/<embedder_id>/`.
Each (chunker × src) gets its own collection inside the embedder's DB.

Why one DB per embedder, not one DB total: different embedders produce
different-dim vectors and live in different vector spaces. Mixing them in
the same collection breaks retrieval.

Embedder registry:

  Base models (E1-E9 from docs/experiments.md):
    e1  intfloat/multilingual-e5-large       multilingual, 1024-dim, e5-prefix
    e2  intfloat/multilingual-e5-base        multilingual, 768-dim,  e5-prefix
    e3  sentence-transformers/all-MiniLM-L6-v2  English-only, 384-dim
    e4  BAAI/bge-m3                          multilingual, 1024-dim
    e5  BAAI/bge-large-en-v1.5               English, 1024-dim, BGE-prefix
    e6  jinaai/jina-embeddings-v3            multilingual, 1024-dim
    e7  openai/text-embedding-3-large        API, 3072-dim, Matryoshka-capable
    e8  openai/text-embedding-3-small        API, 1536-dim, Matryoshka-capable
    e9  cohere/embed-multilingual-v3.0       API, 1024-dim

  Advanced techniques (X-series from docs/advanced-embedding.md):
    x4  BM25 keyword index (no model, no key)
    x7  BGE-M3 multi-output (dense + sparse via FlagEmbedding)
    x8  Matryoshka — append `--matryoshka-dim N` to any base embedder

  X1 reranker, X2 HyDE, X3 multi-query, X5 prefixes (already done) live in
  retrieve.py — they are query-time, not embed-time.

IMPORTANT (X5): e5 and BGE expect "passage: ..." / "query: ..." prefixes.
The backends below handle this automatically. Skipping prefixes silently
drops precision by 5-15%.

Usage:
    python -m scripts.embed --embedder e2 --chunker c1 --src deep-translator
    python -m scripts.embed --embedder x4 --chunker all-no-key            # BM25
    python -m scripts.embed --embedder e7 --matryoshka-dim 512             # X8
    python -m scripts.embed --embedder x7 --chunker c1                     # BGE-M3 multi
    python -m scripts.embed --embedder all-no-key --chunker c1
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    # --- Advanced (X-series) ---
    "x4": {
        # X4: BM25 keyword index. Not a vector embedder; produces a sparse
        # lexical index that complements dense embedding for hybrid retrieval.
        "kind": "bm25",
        "model": None,
        "dim": None,
        "needs_key": False,
    },
    "x7": {
        # X7: BGE-M3 multi-output (dense + sparse + ColBERT in one forward pass).
        # We persist dense (Chroma) and sparse (JSONL); ColBERT is skipped
        # because Chroma doesn't support multi-vector storage.
        "kind": "bge-m3-multi",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "needs_key": False,
        "needs_lib": "FlagEmbedding",
    },
}


def embedder_id(eid: str, matryoshka_dim: int | None = None) -> str:
    """Folder-friendly id like 'e1-multilingual-e5-large' or 'x4-bm25'.
    If --matryoshka-dim is set, suffixed with '-d<N>'."""
    info = EMBEDDERS[eid]
    if info.get("model"):
        short = info["model"].split("/")[-1]
        base = f"{eid}-{short}"
    else:
        base = f"{eid}-{info['kind']}"
    if matryoshka_dim:
        base = f"{base}-d{matryoshka_dim}"
    return base


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class STBackend:
    def __init__(self, info: dict, matryoshka_dim: int | None = None):
        from sentence_transformers import SentenceTransformer

        print(f"[embed] loading {info['model']} ...", file=sys.stderr)
        # trust_remote_code needed for jina-v3 and similar
        self.model = SentenceTransformer(info["model"], trust_remote_code=True)
        self.passage_prefix = info.get("passage_prefix", "")
        self.query_prefix = info.get("query_prefix", "")
        self.dim = matryoshka_dim or info["dim"]
        self.matryoshka_dim = matryoshka_dim

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        prefixed = [self.passage_prefix + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        if self.matryoshka_dim:
            # X8: truncate and re-normalize. Note: only models trained with
            # Matryoshka loss produce valid truncations. Use with caution on
            # non-Matryoshka models.
            vecs = vecs[:, : self.matryoshka_dim]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs.tolist()


class OpenAIBackend:
    BATCH = 100  # OpenAI accepts up to 2048 inputs per call; keep it modest.

    def __init__(self, info: dict, matryoshka_dim: int | None = None):
        if not os.getenv(info["env_var"]):
            raise SystemExit(f"{info['env_var']} not set — required for {info['model']}.")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = info["model"]
        self.dim = matryoshka_dim or info["dim"]
        self.matryoshka_dim = matryoshka_dim

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self.model}
        if self.matryoshka_dim:
            # X8: OpenAI 3-large/3-small support `dimensions` natively; the
            # API returns vectors already truncated and re-normalized.
            kwargs["dimensions"] = self.matryoshka_dim
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            resp = self.client.embeddings.create(input=batch, **kwargs)
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


class BM25Backend:
    """X4: keyword-based BM25 index. Builds an inverted index per
    (chunker, src) collection and saves it as a pickle file. Not stored in
    Chroma — Chroma is for dense vectors. At retrieval time, retrieve.py
    loads this index and runs BM25 score against the query."""

    TOKEN_RE = re.compile(r"\w+")

    def __init__(self, info: dict, matryoshka_dim: int | None = None):
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError as e:
            raise SystemExit("Install rank-bm25: pip install rank-bm25") from e
        self.dim = None  # not a dense embedder
        if matryoshka_dim:
            print("[embed] note: --matryoshka-dim ignored for BM25", file=sys.stderr)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls.TOKEN_RE.findall(text.lower())

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        # Required by the dense pipeline; BM25 doesn't use this.
        raise NotImplementedError("BM25 builds a single index, not per-passage vectors")


class BGEM3MultiBackend:
    """X7: BGE-M3 multi-output. Single forward pass produces dense + sparse
    (lexical_weights) + ColBERT (token-level) outputs. We persist:
      - dense  → Chroma (same shape as E4, but written to a separate DB)
      - sparse → JSONL alongside the Chroma DB
      - ColBERT → SKIPPED for MVP (Chroma can't store multi-vec)
    """

    def __init__(self, info: dict, matryoshka_dim: int | None = None):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise SystemExit(
                "Install FlagEmbedding: pip install -U FlagEmbedding"
            ) from e
        print(f"[embed] loading BGE-M3 (multi-output) ...", file=sys.stderr)
        self.model = BGEM3FlagModel(info["model"], use_fp16=True)
        self.dim = matryoshka_dim or info["dim"]
        self.matryoshka_dim = matryoshka_dim
        # Buffer the most recent encode call so we can write sparse alongside
        # the dense vectors written through the standard Chroma path.
        self._last_sparse: list[dict[str, float]] | None = None

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        out = self.model.encode(
            texts,
            batch_size=8,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        sparse = out.get("lexical_weights") or []
        if self.matryoshka_dim:
            dense = dense[:, : self.matryoshka_dim]
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            dense = dense / np.clip(norms, 1e-12, None)
        # FlagEmbedding sparse output: list of dict[token_id_str -> weight]
        # Stash for downstream sparse-JSONL writer.
        self._last_sparse = [dict(s) for s in sparse]
        return dense.tolist()

    def take_last_sparse(self) -> list[dict[str, float]]:
        s = self._last_sparse or []
        self._last_sparse = None
        return s


def make_backend(eid: str, matryoshka_dim: int | None = None):
    info = EMBEDDERS[eid]
    kind = info["kind"]
    if kind == "sentence-transformers":
        return STBackend(info, matryoshka_dim=matryoshka_dim)
    if kind == "openai":
        return OpenAIBackend(info, matryoshka_dim=matryoshka_dim)
    if kind == "cohere":
        if matryoshka_dim:
            print("[embed] note: --matryoshka-dim ignored for Cohere", file=sys.stderr)
        return CohereBackend(info)
    if kind == "bm25":
        return BM25Backend(info, matryoshka_dim=matryoshka_dim)
    if kind == "bge-m3-multi":
        return BGEM3MultiBackend(info, matryoshka_dim=matryoshka_dim)
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
    matryoshka_dim: int | None = None,
    batch_size: int = 256,
) -> dict:
    info = EMBEDDERS[eid]
    kind = info["kind"]
    eid_full = embedder_id(eid, matryoshka_dim=matryoshka_dim)

    if kind == "bm25":
        return _embed_one_bm25(eid_full, chunker_folder, chunker_dir, src, force)
    if kind == "bge-m3-multi":
        return _embed_one_bge_m3_multi(
            eid_full, backend, chunker_folder, chunker_dir, src, force, batch_size
        )
    return _embed_one_dense(
        eid_full, backend, chunker_folder, chunker_dir, src, force, batch_size
    )


def _embed_one_dense(
    eid_full, backend, chunker_folder, chunker_dir, src, force, batch_size
) -> dict:
    """Standard dense-vector path: write each chunk's embedding to Chroma."""
    import chromadb

    db_path = VECTOR_DB_BASE / eid_full
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    coll_name = collection_name(chunker_folder, src)
    coll = client.get_or_create_collection(name=coll_name)

    chunks = list(load_chunks_from_dir(chunker_dir))
    n_total = len(chunks)
    existing = coll.count()
    if existing >= n_total and not force:
        return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
                "collection": coll_name, "chunks_in_jsonl": n_total,
                "chunks_in_chroma": existing, "skipped_existing": True,
                "elapsed_seconds": 0.0}
    if force:
        client.delete_collection(coll_name)
        coll = client.get_or_create_collection(name=coll_name)

    t0 = time.time()
    n_added = 0
    for start in range(0, n_total, batch_size):
        batch = chunks[start : start + batch_size]
        vecs = backend.encode_passages([c["text"] for c in batch])
        coll.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vecs,
            metadatas=[sanitize_metadata(c["metadata"]) for c in batch],
        )
        n_added += len(batch)
    return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
            "collection": coll_name, "chunks_in_jsonl": n_total,
            "chunks_added": n_added, "skipped_existing": False,
            "elapsed_seconds": round(time.time() - t0, 2)}


def _embed_one_bm25(eid_full, chunker_folder, chunker_dir, src, force) -> dict:
    """X4 storage layout:
        vector_db/x4-bm25/<chunker>__<src>/
            ├── index.pkl    pickled BM25Okapi
            └── docs.jsonl   id,text,metadata per chunk (for retrieval lookup)
    """
    import pickle
    from rank_bm25 import BM25Okapi

    save_dir = VECTOR_DB_BASE / eid_full / collection_name(chunker_folder, src)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "index.pkl").exists() and not force:
        return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
                "collection": save_dir.name, "skipped_existing": True,
                "elapsed_seconds": 0.0}

    chunks = list(load_chunks_from_dir(chunker_dir))
    if not chunks:
        return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
                "collection": save_dir.name, "chunks_in_jsonl": 0,
                "skipped_existing": False, "elapsed_seconds": 0.0}

    t0 = time.time()
    tokens = [BM25Backend.tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokens)
    with (save_dir / "index.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    with (save_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c["id"],
                "text": c["text"],
                "metadata": sanitize_metadata(c["metadata"]),
            }, ensure_ascii=False) + "\n")
    return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
            "collection": save_dir.name, "chunks_in_jsonl": len(chunks),
            "chunks_added": len(chunks), "skipped_existing": False,
            "elapsed_seconds": round(time.time() - t0, 2)}


def _embed_one_bge_m3_multi(
    eid_full, backend, chunker_folder, chunker_dir, src, force, batch_size
) -> dict:
    """X7 storage layout:
        vector_db/x7-bge-m3/<chroma_dir>     dense vectors in Chroma (collection)
        vector_db/x7-bge-m3/sparse/<coll>.jsonl   sparse weights per chunk
    """
    import chromadb

    db_path = VECTOR_DB_BASE / eid_full
    db_path.mkdir(parents=True, exist_ok=True)
    sparse_dir = db_path / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    coll_name = collection_name(chunker_folder, src)
    coll = client.get_or_create_collection(name=coll_name)

    chunks = list(load_chunks_from_dir(chunker_dir))
    n_total = len(chunks)
    sparse_path = sparse_dir / f"{coll_name}.jsonl"
    if coll.count() >= n_total and sparse_path.exists() and not force:
        return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
                "collection": coll_name, "chunks_in_jsonl": n_total,
                "chunks_in_chroma": coll.count(), "skipped_existing": True,
                "elapsed_seconds": 0.0}
    if force:
        client.delete_collection(coll_name)
        coll = client.get_or_create_collection(name=coll_name)
        if sparse_path.exists():
            sparse_path.unlink()

    t0 = time.time()
    n_added = 0
    with sparse_path.open("w", encoding="utf-8") as sf:
        for start in range(0, n_total, batch_size):
            batch = chunks[start : start + batch_size]
            dense = backend.encode_passages([c["text"] for c in batch])
            sparse = backend.take_last_sparse()
            coll.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=dense,
                metadatas=[sanitize_metadata(c["metadata"]) for c in batch],
            )
            for c, s in zip(batch, sparse):
                sf.write(json.dumps({"id": c["id"], "sparse": s},
                                    ensure_ascii=False) + "\n")
            n_added += len(batch)
    return {"embedder": eid_full, "chunker": chunker_folder, "src": src,
            "collection": coll_name, "chunks_in_jsonl": n_total,
            "chunks_added": n_added, "skipped_existing": False,
            "elapsed_seconds": round(time.time() - t0, 2)}


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
    if arg == "all-dense-no-key":
        # exclude BM25 since it isn't a dense embedder
        return [e for e, info in EMBEDDERS.items()
                if not info["needs_key"] and info["kind"] != "bm25"]
    if arg == "all-advanced":
        # X-series only
        return [e for e in EMBEDDERS if e.startswith("x")]
    if arg in EMBEDDERS:
        return [arg]
    raise SystemExit(f"unknown embedder: {arg!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedder", required=True,
        help=f"e1..e9, x4 (BM25), x7 (BGE-M3 multi-output), 'all', "
             f"'all-no-key', 'all-dense-no-key' (E1-E6+x7), or 'all-advanced' "
             f"(x4+x7). Available: {sorted(EMBEDDERS)}",
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
    ap.add_argument("--matryoshka-dim", type=int, default=None,
                    help="X8: truncate output vectors to this many dims and "
                         "re-normalize. Suffixes embedder dir with -d<N>. Use only "
                         "with Matryoshka-trained models (OpenAI 3-large/3-small).")
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
        backend = make_backend(eid, matryoshka_dim=args.matryoshka_dim)
        for src in src_options:
            targets = expand_chunker_arg(args.chunker, src)
            for chunker_folder, chunker_dir in targets:
                try:
                    info = embed_one(
                        eid, backend, chunker_folder, chunker_dir, src,
                        args.force, matryoshka_dim=args.matryoshka_dim,
                    )
                    print(
                        f"[embed] {info.get('embedder', eid)} × {chunker_folder} × {src} → "
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
