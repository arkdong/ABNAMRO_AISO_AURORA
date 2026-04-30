"""Chunking strategies (simple C1-C10 + advanced A1-A8) per docs/.

Reads translated articles from `data/translated/<src>/<slug>.md` (or
`data/raw/writing_guide.md` when `--src writing-guide`) and writes JSONL to
`data/chunked/<tier>/<chunker>/<src>/<slug>.jsonl`. One line per chunk:
  {"id", "text", "metadata": {...}}

Simple methods (article-applicable):
  c1   fixed-size 512/50  (token-based, tiktoken cl100k_base)
  c2   fixed-size 256/25
  c3   fixed-size 1024/100
  c4   paragraph
  c5   sentence-window (each chunk = sentence + 2 neighbours each side)
  c6   semantic (split where adjacent-sentence embedding similarity drops)
  c10  article-as-chunk

Simple methods (writing-guide-only):
  c7   heading
  c8   heading-with-parent (breadcrumb in TEXT)
  c9   sliding-window

Advanced methods:
  a1   contextual retrieval (Anthropic-style; needs LLM key)
  a2   late chunking — chunks are paragraph splits with marker
       `requires_late_chunking: true`; embedding step computes contextual vectors
  a3   proposition-based (LLM extracts atomic claims; needs LLM key)
  a4   RAPTOR — simplified 2-level (paragraph leaves + cluster summaries; needs LLM key)
  a5   small-to-big (no key)
  a6   HyDE-style indexing (questions per chunk; needs LLM key)
  a7   LLM-as-chunker (needs LLM key)
  a8   structure-aware (writing-guide; breadcrumb in METADATA)

Usage:
    python -m scripts.chunk --method c1 --src deep-translator
    python -m scripts.chunk --method a5 --src deep-translator
    python -m scripts.chunk --method a8 --src writing-guide
    python -m scripts.chunk --method a1 --src deep-translator   # needs ANTHROPIC_API_KEY
    python -m scripts.chunk --method all-no-key --src deep-translator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRANSLATED_BASE = ROOT / "data" / "translated"
CHUNKED_BASE = ROOT / "data" / "chunked" / "simple"

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS: dict[str, dict] = {
    "c1":  {"folder": "c1-fixed-512-50",     "applies_to": "article",       "scope": "fast"},
    "c2":  {"folder": "c2-fixed-256-25",     "applies_to": "article",       "scope": "fast"},
    "c3":  {"folder": "c3-fixed-1024-100",   "applies_to": "article",       "scope": "fast"},
    "c4":  {"folder": "c4-paragraph",        "applies_to": "article",       "scope": "fast"},
    "c5":  {"folder": "c5-sentence-window",  "applies_to": "article",       "scope": "fast"},
    "c6":  {"folder": "c6-semantic",         "applies_to": "article",       "scope": "slow"},
    "c7":  {"folder": "c7-heading",          "applies_to": "writing_guide", "scope": "fast"},
    "c8":  {"folder": "c8-heading-with-parent", "applies_to": "writing_guide", "scope": "fast"},
    "c9":  {"folder": "c9-sliding-window",   "applies_to": "writing_guide", "scope": "fast"},
    "c10": {"folder": "c10-article-as-chunk", "applies_to": "article",      "scope": "fast"},
}

# ---------------------------------------------------------------------------
# Chunkers — each returns list[str]
# ---------------------------------------------------------------------------


def chunk_fixed_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive split on token count (cl100k_base via tiktoken)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk_paragraph(text: str, min_len: int = 50) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > min_len]


_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — strips markdown headings/links to plain
    text first, then splits on sentence-final punctuation followed by a capital."""
    # Drop markdown bullet markers but keep the bullet content
    cleaned = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    # Strip ATX heading markers
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # Convert paragraph breaks to sentence-end punctuation if missing
    parts: list[str] = []
    for para in cleaned.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Add period if paragraph doesn't end with sentence punctuation
        if para[-1] not in ".!?":
            para += "."
        parts.append(para)
    joined = " ".join(parts)
    sentences = _SENT_END_RE.split(joined)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentence_window(text: str, n_neighbors: int = 2) -> list[str]:
    """Each chunk = sentence i +/- n_neighbors sentences."""
    sents = _split_sentences(text)
    out: list[str] = []
    for i in range(len(sents)):
        a = max(0, i - n_neighbors)
        b = min(len(sents), i + n_neighbors + 1)
        out.append(" ".join(sents[a:b]))
    return out


# Cached embedder for semantic chunking
_SEM_EMBEDDER = None


def _semantic_embedder():
    global _SEM_EMBEDDER
    if _SEM_EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        print("[c6] loading sentence-transformer (all-MiniLM-L6-v2)...", file=sys.stderr)
        _SEM_EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SEM_EMBEDDER


def chunk_semantic(text: str, breakpoint_percentile: float = 75.0) -> list[str]:
    """Split where cosine similarity between adjacent sentences drops below the
    Pth percentile of all gaps (i.e. the lowest-similarity 25% of boundaries)."""
    import numpy as np

    sents = _split_sentences(text)
    if len(sents) <= 1:
        return [text.strip()] if text.strip() else []

    model = _semantic_embedder()
    vecs = model.encode(sents, normalize_embeddings=True, show_progress_bar=False)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)  # cosine sim between adjacent
    # Lowest sims = strongest topic shift. Below the Pth percentile = breakpoint.
    threshold = float(np.percentile(sims, 100.0 - breakpoint_percentile))
    breakpoints = [i + 1 for i, s in enumerate(sims) if s <= threshold]

    chunks: list[str] = []
    start = 0
    for bp in breakpoints:
        chunks.append(" ".join(sents[start:bp]).strip())
        start = bp
    chunks.append(" ".join(sents[start:]).strip())
    return [c for c in chunks if c]


def chunk_article_as_one(text: str) -> list[str]:
    return [text.strip()] if text.strip() else []


# ---------------------------------------------------------------------------
# Writing-guide heading-aware chunkers (C7, C8, C9)
# ---------------------------------------------------------------------------

# Strict numbered-heading detector for Writing Guide:
#   chapters [1-6], sections X.Y, subsections X.Y.Z, only on their own line,
#   title must start with capital and contain no inner digits (filters pypdf's
#   "2.1 Title 2.2 Title" merged-line artifacts).
WG_HEADING_RE = re.compile(
    r"^(?P<num>[1-6]\.(?:\d+(?:\.\d+)?)?)\s+(?P<title>[A-Z][^\n\d]{2,100}?)\s*$",
    re.MULTILINE,
)

# Whitelist of valid Writing Guide chapter headings. The PDF has FAQ-style
# numbered lists ("1. Why...", "2. What are...") that match the heading regex
# but aren't real chapters. Filtering by first word of expected title removes
# them. Update if the Writing Guide structure changes.
WG_EXPECTED_CHAPTER_PREFIX: dict[int, str] = {
    1: "Prepare",
    2: "Structure",
    3: "Wording",
    4: "Accuracy",
    5: "Punctuation",
    6: "Inclusive",
}


def _wg_split_by_heading(text: str) -> list[dict]:
    """Return a list of {num, title, body, parent_chapter, parent_section}.
    The first segment (text before any heading) is included with num=None."""
    raw = list(WG_HEADING_RE.finditer(text))
    matches = []
    seen_chapters: set[int] = set()
    for m in raw:
        num = m.group("num").rstrip(".")
        title = m.group("title").strip()
        parts = [int(p) for p in num.split(".")]
        if len(parts) == 1:
            ch = parts[0]
            if ch in seen_chapters:
                continue
            expected = WG_EXPECTED_CHAPTER_PREFIX.get(ch)
            if not expected or not title.startswith(expected):
                continue  # FAQ false positive or body-text false positive
            seen_chapters.add(ch)
        matches.append(m)
    segments: list[dict] = []

    # preamble before first heading
    if matches and matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            segments.append(
                {"num": None, "title": "Preamble", "body": pre,
                 "parent_chapter": None, "parent_section": None}
            )

    parent_chapter: tuple[str, str] | None = None
    parent_section: tuple[str, str] | None = None

    for i, m in enumerate(matches):
        num = m.group("num").rstrip(".")
        title = m.group("title").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        depth = num.count(".") + 1  # 1 -> chapter, 1.1 -> section, 1.1.1 -> subsection

        # Track parents BEFORE recording (so a subsection sees its parent section)
        if depth == 1:
            parent_chapter = (num, title)
            parent_section = None
        elif depth == 2:
            parent_section = (num, title)
            # parent_chapter stays from last chapter
        # depth 3 inherits both

        segments.append(
            {
                "num": num,
                "title": title,
                "body": body,
                "depth": depth,
                "parent_chapter": parent_chapter,
                "parent_section": parent_section if depth == 3 else None,
            }
        )
    return segments


def chunk_heading(text: str) -> list[str]:
    """C7: each chunk is a single section's content, prefixed with its heading."""
    out: list[str] = []
    for seg in _wg_split_by_heading(text):
        if not seg["body"]:
            continue
        if seg["num"]:
            out.append(f"{seg['num']} {seg['title']}\n\n{seg['body']}")
        else:
            out.append(seg["body"])
    return out


def chunk_heading_with_parent(text: str) -> list[str]:
    """C8: same chunks as C7, but each chunk's TEXT prepends a breadcrumb path
    so the embedding sees the parent context."""
    out: list[str] = []
    for seg in _wg_split_by_heading(text):
        if not seg["body"]:
            continue
        crumbs: list[str] = []
        if seg.get("parent_chapter"):
            n, t = seg["parent_chapter"]
            crumbs.append(f"{n} {t}")
        if seg.get("parent_section"):
            n, t = seg["parent_section"]
            crumbs.append(f"{n} {t}")
        if seg["num"]:
            crumbs.append(f"{seg['num']} {seg['title']}")
        breadcrumb = " > ".join(crumbs) if crumbs else "Preamble"
        out.append(f"[Breadcrumb: {breadcrumb}]\n\n{seg['body']}")
    return out


def chunk_sliding_window(text: str, window: int = 2, stride: int = 1) -> list[str]:
    """C9: sliding window over heading-segments. Each chunk = `window` consecutive
    sections joined together. Catches rules that span sections."""
    segs = _wg_split_by_heading(text)
    # represent each segment as (heading + body) text
    blocks: list[str] = []
    for seg in segs:
        if not seg["body"]:
            continue
        if seg["num"]:
            blocks.append(f"{seg['num']} {seg['title']}\n\n{seg['body']}")
        else:
            blocks.append(seg["body"])
    out: list[str] = []
    for i in range(0, max(1, len(blocks) - window + 1), stride):
        out.append("\n\n".join(blocks[i : i + window]))
    return out


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


def get_chunker(method: str) -> Callable[[str], list[str]]:
    if method == "c1":  return lambda t: chunk_fixed_size(t, 512, 50)
    if method == "c2":  return lambda t: chunk_fixed_size(t, 256, 25)
    if method == "c3":  return lambda t: chunk_fixed_size(t, 1024, 100)
    if method == "c4":  return chunk_paragraph
    if method == "c5":  return chunk_sentence_window
    if method == "c6":  return chunk_semantic
    if method == "c7":  return chunk_heading
    if method == "c8":  return chunk_heading_with_parent
    if method == "c9":  return chunk_sliding_window
    if method == "c10": return chunk_article_as_one
    raise SystemExit(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Frontmatter I/O
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def read_md(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: no frontmatter")
    fm = yaml.safe_load(m.group(1)) or {}
    body = text[m.end():].lstrip("\n")
    return fm, body


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


def make_metadata(
    *,
    article_fm: dict,
    chunker_id: str,
    src_method: str,
    chunk_index: int,
    n_chunks_total: int,
    slug: str,
) -> dict:
    doc_type = article_fm.get("doc_type", "article")
    return {
        "chunk_id": f"{slug}__{chunker_id}__{chunk_index:04d}",
        "source_slug": slug,
        "source_title": article_fm.get("title"),
        "source_url": article_fm.get("source_url") or article_fm.get("url"),
        "source_date": article_fm.get("date"),
        "doc_type": doc_type,
        "channel": "website",
        "content_type": "insight_article" if doc_type == "article" else None,
        "sector": article_fm.get("sector"),
        "topic": article_fm.get("topic"),
        "language_original": article_fm.get("language_original"),
        "language_embedded": article_fm.get("language", "en"),
        "translation_method": src_method if doc_type == "article" else None,
        "translated_with": article_fm.get("translated_with"),
        "chunker": chunker_id,
        "chunk_index": chunk_index,
        "n_chunks_total": n_chunks_total,
        "status": "approved",
        "date_ingested": date.today().isoformat(),
    }


def process_article(
    src_path: Path,
    out_dir: Path,
    chunker: Callable[[str], list[str]],
    chunker_id: str,
    src_method: str,
) -> tuple[int, int]:
    """Chunk one article, write JSONL, return (n_chunks, total_chars)."""
    fm, body = read_md(src_path)
    chunks = chunker(body)
    if not chunks:
        return 0, 0

    slug = src_path.stem
    out_path = out_dir / f"{slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(chunks):
            meta = make_metadata(
                article_fm=fm,
                chunker_id=chunker_id,
                src_method=src_method,
                chunk_index=i,
                n_chunks_total=len(chunks),
                slug=slug,
            )
            row = {"id": meta["chunk_id"], "text": text, "metadata": meta}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_chars += len(text)
    return len(chunks), total_chars


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(method: str, src_method: str, limit: int | None, force: bool) -> None:
    if method not in METHODS:
        raise SystemExit(f"unknown method: {method!r}")
    info = METHODS[method]
    chunker_folder = info["folder"]

    # Resolve input source: writing-guide special case vs article translations
    if src_method == "writing-guide":
        if info["applies_to"] != "writing_guide":
            raise SystemExit(
                f"{method} is for {info['applies_to']!r}; use src=deep-translator (or other translation method)"
            )
        wg_path = ROOT / "data" / "raw" / "writing_guide.md"
        if not wg_path.exists():
            raise SystemExit(f"{wg_path} not found — run scripts/ingest_pdf.py first")
        files = [wg_path]
    else:
        if info["applies_to"] != "article":
            raise SystemExit(
                f"{method} is for {info['applies_to']!r}; use src=writing-guide"
            )
        src_dir = TRANSLATED_BASE / src_method
        if not src_dir.exists():
            raise SystemExit(f"source dir not found: {src_dir}")
        files = sorted(src_dir.glob("*.md"))

    out_dir = CHUNKED_BASE / chunker_folder / src_method
    out_dir.mkdir(parents=True, exist_ok=True)

    chunker = get_chunker(method)
    print(
        f"[chunk] method={method} ({chunker_folder}) src={src_method} "
        f"in={len(files)} out={out_dir}",
        file=sys.stderr,
    )

    t0 = time.time()
    done = 0
    skipped = 0
    n_chunks_total = 0
    n_chars_total = 0
    for src in files:
        if limit and done >= limit:
            break
        out_path = out_dir / f"{src.stem}.jsonl"
        if out_path.exists() and not force:
            skipped += 1
            continue
        n, chars = process_article(src, out_dir, chunker, chunker_folder, src_method)
        done += 1
        n_chunks_total += n
        n_chars_total += chars

    summary = {
        "method": method,
        "chunker_folder": chunker_folder,
        "src_method": src_method,
        "documents_processed": done,
        "documents_skipped_existing": skipped,
        "chunks_total": n_chunks_total,
        "chars_total": n_chars_total,
        "chars_per_chunk_avg": round(n_chars_total / n_chunks_total, 1) if n_chunks_total else 0,
        "chunks_per_doc_avg": round(n_chunks_total / done, 2) if done else 0,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), file=sys.stderr)


def main() -> None:
    fast_methods = [m for m, info in METHODS.items()
                    if info["applies_to"] == "article" and info["scope"] == "fast"]
    article_methods = [m for m, info in METHODS.items() if info["applies_to"] == "article"]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        help=f"chunking method: one of {sorted(METHODS)}, "
             f"'all-fast' ({fast_methods}), or 'all-article' ({article_methods})",
    )
    ap.add_argument("--src", default="deep-translator", help="translation method subfolder")
    ap.add_argument("--limit", type=int, default=None, help="cap articles for testing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.method == "all-fast":
        targets = fast_methods
    elif args.method == "all-article":
        targets = article_methods
    else:
        targets = [args.method]

    for m in targets:
        run(m, args.src, args.limit, args.force)


if __name__ == "__main__":
    main()
