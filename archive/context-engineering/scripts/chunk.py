"""Chunking strategies (simple C1-C10 + advanced A1-A8) per docs/.

Reads translated articles from the shared repo-root `data/article/<lang>/`
(or `data/writing_guide.md` when `--src writing-guide`) and writes JSONL to
`chunked/<tier>/<chunker>/<src>/<slug>.jsonl`. One line per chunk:
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

ROOT = Path(__file__).resolve().parent.parent           # archive/context-engineering/
ARCHIVE_ROOT = ROOT.parent
PROJECT_ROOT = ARCHIVE_ROOT.parent

# Article sources live at <repo-root>/data/article/{en,nl}/. Map our familiar
# `--src` names to those paths. Add new entries here when more translation
# methods land (e.g. anthropic-claude-haiku-4-5 → article/anthropic-haiku/).
TRANSLATED_PATHS: dict[str, Path] = {
    "gpt-5":     PROJECT_ROOT / "data" / "article" / "en",
    "source-nl": PROJECT_ROOT / "data" / "article" / "nl",
}

CHUNKED_BASE = ROOT / "chunked"                          # was data/chunked/

# ---------------------------------------------------------------------------
# Method registry — tier ∈ {simple, advanced}, scope ∈ {fast, slow, llm, llm-slow}
# ---------------------------------------------------------------------------

METHODS: dict[str, dict] = {
    # Simple
    "c1":  {"tier": "simple",   "folder": "c1-fixed-512-50",       "applies_to": "article",       "scope": "fast"},
    "c2":  {"tier": "simple",   "folder": "c2-fixed-256-25",       "applies_to": "article",       "scope": "fast"},
    "c3":  {"tier": "simple",   "folder": "c3-fixed-1024-100",     "applies_to": "article",       "scope": "fast"},
    "c4":  {"tier": "simple",   "folder": "c4-paragraph",          "applies_to": "article",       "scope": "fast"},
    "c5":  {"tier": "simple",   "folder": "c5-sentence-window",    "applies_to": "article",       "scope": "fast"},
    "c6":  {"tier": "simple",   "folder": "c6-semantic",           "applies_to": "article",       "scope": "slow"},
    "c7":  {"tier": "simple",   "folder": "c7-heading",            "applies_to": "writing_guide", "scope": "fast"},
    "c8":  {"tier": "simple",   "folder": "c8-heading-with-parent", "applies_to": "writing_guide", "scope": "fast"},
    "c9":  {"tier": "simple",   "folder": "c9-sliding-window",     "applies_to": "writing_guide", "scope": "fast"},
    "c10": {"tier": "simple",   "folder": "c10-article-as-chunk",  "applies_to": "article",       "scope": "fast"},
    "c11": {"tier": "simple",   "folder": "c11-md-h2-split",       "applies_to": "writing_guide", "scope": "fast"},
    # Advanced
    "a1":  {"tier": "advanced", "folder": "a1-contextual-retrieval", "applies_to": "article",     "scope": "llm"},
    "a2":  {"tier": "advanced", "folder": "a2-late-chunking",        "applies_to": "article",     "scope": "fast"},
    "a3":  {"tier": "advanced", "folder": "a3-proposition-based",    "applies_to": "article",     "scope": "llm"},
    "a4":  {"tier": "advanced", "folder": "a4-raptor",               "applies_to": "article",     "scope": "llm-slow"},
    "a5":  {"tier": "advanced", "folder": "a5-small-to-big",         "applies_to": "article",     "scope": "fast"},
    "a6":  {"tier": "advanced", "folder": "a6-hyde-indexing",        "applies_to": "article",     "scope": "llm"},
    "a7":  {"tier": "advanced", "folder": "a7-llm-as-chunker",       "applies_to": "article",     "scope": "llm"},
    "a8":  {"tier": "advanced", "folder": "a8-structure-aware",      "applies_to": "writing_guide", "scope": "fast"},
    "a9":  {"tier": "advanced", "folder": "a9-hybrid-small-to-big",   "applies_to": "article",     "scope": "slow"},
    "a10": {"tier": "advanced", "folder": "a10-raptor-structural",    "applies_to": "writing_guide", "scope": "fast"},
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


# ---------------------------------------------------------------------------
# Advanced chunkers (A1-A8)
# ---------------------------------------------------------------------------


# Default LLM model for chunking-time calls. Cheap & fast.
LLM_DEFAULT_MODEL_ANTHROPIC = "claude-haiku-4-5"
LLM_DEFAULT_MODEL_OPENAI = "gpt-4o-mini"

_anthropic_client = None
_openai_client = None


def _get_anthropic_client():
    """Lazy-load Anthropic client. Raises if ANTHROPIC_API_KEY missing."""
    global _anthropic_client
    if _anthropic_client is None:
        import os
        from anthropic import Anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set — required for this chunker.")
        _anthropic_client = Anthropic()
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import os
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set — required for this chunker.")
        _openai_client = OpenAI()
    return _openai_client


# --- A1: Contextual retrieval (Anthropic-style) -----------------------------

A1_SYSTEM_PROMPT = """You generate a short context blurb that situates a chunk
within its source document, to improve retrieval quality. Output ONLY the
blurb (50-100 words). No preamble, no quotation marks."""


def chunk_contextual_retrieval(
    text: str, *, llm_provider: str = "anthropic", model: str | None = None
) -> list[dict]:
    """A1: take base C1 chunks, prepend an LLM-generated context blurb to each.
    Anthropic prompt-caching makes the doc-cache hit on every chunk after the
    first, dropping cost ~10x for documents reused across many chunks."""
    base = chunk_fixed_size(text, 512, 50)
    out: list[dict] = []
    if llm_provider == "anthropic":
        client = _get_anthropic_client()
        m = model or LLM_DEFAULT_MODEL_ANTHROPIC
        for chunk in base:
            msg = client.messages.create(
                model=m,
                max_tokens=200,
                system=A1_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{text}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"<chunk>\n{chunk}\n</chunk>\n\n"
                                "Provide a 50-100 word context to situate this chunk within "
                                "the document for retrieval purposes."
                            ),
                        },
                    ],
                }],
            )
            ctx = "".join(b.text for b in msg.content if b.type == "text").strip()
            out.append({"text": f"{ctx}\n\n{chunk}", "context_prefix": ctx, "base_chunk": chunk})
    elif llm_provider == "openai":
        client = _get_openai_client()
        m = model or LLM_DEFAULT_MODEL_OPENAI
        for chunk in base:
            resp = client.chat.completions.create(
                model=m,
                temperature=0,
                messages=[
                    {"role": "system", "content": A1_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"<document>\n{text}\n</document>\n\n"
                        f"<chunk>\n{chunk}\n</chunk>\n\n"
                        "Provide a 50-100 word context to situate this chunk within the "
                        "document for retrieval purposes."
                    )},
                ],
            )
            ctx = (resp.choices[0].message.content or "").strip()
            out.append({"text": f"{ctx}\n\n{chunk}", "context_prefix": ctx, "base_chunk": chunk})
    else:
        raise SystemExit(f"unknown llm_provider: {llm_provider!r}")
    return out


# --- A2: Late chunking (chunks identical to C4 paragraph; flag for embed) ---


def chunk_late(text: str) -> list[dict]:
    """A2: produce paragraph chunks marked for late-chunking embedding. The
    actual contextual embedding happens at embed time — this stage only marks
    the chunks so the embedder knows to feed the whole-document context."""
    paragraphs = chunk_paragraph(text)
    return [
        {"text": p, "requires_late_chunking": True, "source_doc_offset": idx}
        for idx, p in enumerate(paragraphs)
    ]


# --- A3: Proposition-based ---------------------------------------------------

A3_SYSTEM_PROMPT = """You decompose text into atomic, self-contained propositions.

Rules:
- Each proposition is a single declarative claim, ~10-30 words.
- No coreference: replace pronouns with the entities they refer to.
- Preserve facts (names, numbers, dates) verbatim.
- Output a JSON array of strings. No preamble. No commentary. No code fences."""


def _parse_json_list(raw: str) -> list[str]:
    """Tolerate code fences and extra prose around a JSON list response."""
    raw = raw.strip()
    # strip ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # find first [ ... ]
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [str(x) for x in data if x]
    except json.JSONDecodeError:
        return []


def chunk_propositions(
    text: str, *, llm_provider: str = "anthropic", model: str | None = None
) -> list[dict]:
    """A3: LLM extracts atomic propositions; each proposition becomes a chunk."""
    if llm_provider == "anthropic":
        client = _get_anthropic_client()
        m = model or LLM_DEFAULT_MODEL_ANTHROPIC
        msg = client.messages.create(
            model=m,
            max_tokens=4096,
            system=A3_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text")
    elif llm_provider == "openai":
        client = _get_openai_client()
        m = model or LLM_DEFAULT_MODEL_OPENAI
        resp = client.chat.completions.create(
            model=m,
            temperature=0,
            messages=[
                {"role": "system", "content": A3_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content or ""
    else:
        raise SystemExit(f"unknown llm_provider: {llm_provider!r}")
    propositions = _parse_json_list(raw)
    return [{"text": p} for p in propositions]


# --- A4: RAPTOR (simplified 2-level: paragraph leaves + cluster summaries) --

A4_CLUSTER_SUMMARY_PROMPT = """Summarise the following passages into one
self-contained paragraph (3-6 sentences) capturing their shared theme.
Output ONLY the summary. No preamble."""


def chunk_raptor(
    text: str, *, llm_provider: str = "anthropic", model: str | None = None,
    n_clusters: int | None = None,
) -> list[dict]:
    """A4: simplified RAPTOR with 2 levels.
       L0 = paragraph chunks (leaves)
       L1 = LLM-generated summaries, one per k-means cluster of L0 chunks.
    Both levels are returned as chunks; metadata.tree_level distinguishes them.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    leaves = chunk_paragraph(text)
    if len(leaves) <= 2:
        return [{"text": l, "tree_level": 0, "cluster_id": None} for l in leaves]

    embedder = _semantic_embedder()
    leaf_vecs = embedder.encode(leaves, normalize_embeddings=True, show_progress_bar=False)
    k = n_clusters or max(2, min(8, int(np.ceil(np.sqrt(len(leaves))))))
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(leaf_vecs)

    out: list[dict] = []
    for i, (l, cid) in enumerate(zip(leaves, km.labels_)):
        out.append({"text": l, "tree_level": 0, "cluster_id": int(cid)})

    # summarise each cluster via LLM
    if llm_provider == "anthropic":
        client = _get_anthropic_client()
        m = model or LLM_DEFAULT_MODEL_ANTHROPIC
        def summarise(passages: list[str]) -> str:
            joined = "\n\n---\n\n".join(passages)
            msg = client.messages.create(
                model=m, max_tokens=400, system=A4_CLUSTER_SUMMARY_PROMPT,
                messages=[{"role": "user", "content": joined}],
            )
            return "".join(b.text for b in msg.content if b.type == "text").strip()
    elif llm_provider == "openai":
        client = _get_openai_client()
        m = model or LLM_DEFAULT_MODEL_OPENAI
        def summarise(passages: list[str]) -> str:
            joined = "\n\n---\n\n".join(passages)
            resp = client.chat.completions.create(
                model=m, temperature=0,
                messages=[
                    {"role": "system", "content": A4_CLUSTER_SUMMARY_PROMPT},
                    {"role": "user", "content": joined},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
    else:
        raise SystemExit(f"unknown llm_provider: {llm_provider!r}")

    for cid in range(k):
        cluster_leaves = [l for l, c in zip(leaves, km.labels_) if c == cid]
        if not cluster_leaves:
            continue
        summary = summarise(cluster_leaves)
        out.append({"text": summary, "tree_level": 1, "cluster_id": int(cid),
                    "n_leaves": len(cluster_leaves)})
    return out


# --- A5: Small-to-big -------------------------------------------------------


def chunk_small_to_big(
    text: str, child_size: int = 256, child_overlap: int = 25,
    parent_size: int = 1024, parent_overlap: int = 100,
) -> list[dict]:
    """A5: produce small (child) chunks for indexing, with the larger parent
    chunk attached as metadata. At retrieval time, match on small, return big."""
    parents = chunk_fixed_size(text, parent_size, parent_overlap)
    out: list[dict] = []
    for pi, parent in enumerate(parents):
        children = chunk_fixed_size(parent, child_size, child_overlap)
        for ci, child in enumerate(children):
            out.append({
                "text": child,
                "parent_text": parent,
                "parent_index": pi,
                "child_index": ci,
            })
    return out


# --- A6: HyDE-style indexing ------------------------------------------------

A6_SYSTEM_PROMPT = """Given a passage, generate 3-5 questions a reader might
ask that the passage would answer. Each question should be standalone and
specific.

Output a JSON array of strings. No preamble, no code fences, no commentary."""


def chunk_hyde(
    text: str, *, llm_provider: str = "anthropic", model: str | None = None,
) -> list[dict]:
    """A6: for each base C1 chunk, generate hypothetical questions; each
    question becomes a chunk where text=question and metadata.answer_text=chunk."""
    base = chunk_fixed_size(text, 512, 50)
    out: list[dict] = []
    if llm_provider == "anthropic":
        client = _get_anthropic_client()
        m = model or LLM_DEFAULT_MODEL_ANTHROPIC
        for chunk in base:
            msg = client.messages.create(
                model=m, max_tokens=500, system=A6_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": chunk}],
            )
            raw = "".join(b.text for b in msg.content if b.type == "text")
            for q in _parse_json_list(raw):
                out.append({"text": q, "answer_text": chunk, "is_hyde_question": True})
    elif llm_provider == "openai":
        client = _get_openai_client()
        m = model or LLM_DEFAULT_MODEL_OPENAI
        for chunk in base:
            resp = client.chat.completions.create(
                model=m, temperature=0,
                messages=[
                    {"role": "system", "content": A6_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
            )
            raw = resp.choices[0].message.content or ""
            for q in _parse_json_list(raw):
                out.append({"text": q, "answer_text": chunk, "is_hyde_question": True})
    else:
        raise SystemExit(f"unknown llm_provider: {llm_provider!r}")
    return out


# --- A7: LLM-as-chunker -----------------------------------------------------

A7_SYSTEM_PROMPT = """Split the document into 3-15 semantically coherent
chunks. Each chunk should be 200-500 tokens and represent one cohesive idea
or topic. Do not omit any content.

Output a JSON array of objects with keys:
  - "rationale": one-sentence reason this is a coherent chunk
  - "text": the chunk's text verbatim from the source

No preamble, no code fences, no commentary outside the JSON."""


def chunk_llm_as_chunker(
    text: str, *, llm_provider: str = "anthropic", model: str | None = None,
) -> list[dict]:
    """A7: ask LLM to split the document at semantically coherent boundaries."""
    if llm_provider == "anthropic":
        client = _get_anthropic_client()
        m = model or LLM_DEFAULT_MODEL_ANTHROPIC
        msg = client.messages.create(
            model=m, max_tokens=8192, system=A7_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text")
    elif llm_provider == "openai":
        client = _get_openai_client()
        m = model or LLM_DEFAULT_MODEL_OPENAI
        resp = client.chat.completions.create(
            model=m, temperature=0,
            messages=[
                {"role": "system", "content": A7_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content or ""
    else:
        raise SystemExit(f"unknown llm_provider: {llm_provider!r}")

    # parse JSON array of {rationale, text}
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [{"text": str(x.get("text", "")), "rationale": x.get("rationale")}
            for x in data if x.get("text")]


# --- A9: Hybrid small-to-big (semantic parents + sentence-window children) --


def chunk_hybrid_small_to_big(
    text: str,
    n_neighbors: int = 1,
    breakpoint_percentile: float = 75.0,
) -> list[dict]:
    """A9: Variant 3 of small-to-big.
       - Parents: C6 semantic split (topic-coherent sentence groups).
       - Children inside each parent: sentence-window (focal sentence ± N
         neighbours), one child per sentence position. Children NEVER cut
         mid-sentence.
       - Each child carries its full parent in metadata as parent_text.
    Default window=±1 → each child has 2-3 sentences. Tweak n_neighbors via
    LLM_DEFAULT_MODEL pattern if needed in future.
    """
    import numpy as np

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        only = " ".join(sentences) if sentences else text.strip()
        return [{
            "text": only, "parent_text": only,
            "parent_index": 0, "child_index": 0,
            "n_sentences_in_window": len(sentences),
            "center_sentence_index": 0,
        }] if only else []

    # 1. Find semantic breakpoints (same as C6, but we operate on indices not
    #    on the joined chunk text — we want to keep sentence boundaries).
    model = _semantic_embedder()
    vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    threshold = float(np.percentile(sims, 100.0 - breakpoint_percentile))
    breakpoints = [i + 1 for i, s in enumerate(sims) if s <= threshold]

    # 2. Build parents as lists of sentence-indices.
    parents: list[list[int]] = []
    start = 0
    for bp in breakpoints:
        if bp > start:
            parents.append(list(range(start, bp)))
            start = bp
    if start < len(sentences):
        parents.append(list(range(start, len(sentences))))

    # 3. For each parent, slide a sentence-window over its sentences.
    out: list[dict] = []
    for pi, parent_idx in enumerate(parents):
        parent_text = " ".join(sentences[i] for i in parent_idx).strip()
        if not parent_text:
            continue
        for local_pos, sent_idx in enumerate(parent_idx):
            a = max(0, local_pos - n_neighbors)
            b = min(len(parent_idx), local_pos + n_neighbors + 1)
            window_sentences = [sentences[parent_idx[j]] for j in range(a, b)]
            child_text = " ".join(window_sentences).strip()
            if not child_text:
                continue
            out.append({
                "text": child_text,
                "parent_text": parent_text,
                "parent_index": pi,
                "child_index": local_pos,
                "center_sentence_index": sent_idx,
                "n_sentences_in_window": b - a,
                "n_sentences_in_parent": len(parent_idx),
            })
    return out


# --- A10: RAPTOR-structural (writing guide, no LLM) ------------------------


def chunk_raptor_structural(
    text: str, min_leaf_chars: int = 80, max_leaf_chars: int = 2000,
) -> list[dict]:
    """A10: 3-level tree of (chapter, section, rule) nodes — no LLM.

    Hierarchy is reconstructed from heading prefixes (NOT source order):
      depth 1 — Chapter:   title `N.   Title`     (1 ≤ N ≤ 6)
      depth 2 — Section:   title `N.M Title`     (or `N.M.K`, folded)
      depth 3 — Rule:      anything else; leaves

    Sections are parented to chapter `N` by prefix, so out-of-order PDF
    extraction (sections appearing under the wrong chapter heading) doesn't
    break the tree. Rules are parented by source-order proximity to the most
    recently-seen section/chapter — this is the ONE thing that does need
    sequential processing.

    Each non-leaf node gets a deterministic summary (heading + intro + child
    titles). Each leaf prepends its breadcrumb to the embedded text so the
    embedder sees parental context.
    """
    # Pre-process: split merged headings like "## **3. X** 3.1 Y" into two H2s
    text = re.sub(
        r"(##\s+\**\d+\.\s+[^\n*]+?\**)\s+(\d+\.\d+\s+[^\n]+)$",
        r"\1\n## \2",
        text,
        flags=re.MULTILINE,
    )

    raw = re.split(r"^(?=##\s)", text, flags=re.MULTILINE)
    raw = [r.strip() for r in raw if r.strip() and r.startswith("##")]

    # Patterns
    chap_re = re.compile(r"^##\s+\**(\d+)\.\s+([^*\n]{2,100}?)\**\s*$")
    sect_re = re.compile(r"^##\s+\**(\d+\.\d+(?:\.\d+)?)\s+([^*\n]{2,100}?)\**\s*$")
    rule_re = re.compile(r"^##\s+\**([^*\n]{2,120}?)\**\s*$")

    EXPECTED_CHAPTER_FIRST_WORD = {
        "1": "Prepare", "2": "Structure", "3": "Wording",
        "4": "Accuracy", "5": "Punctuation", "6": "Inclusive",
    }

    def first_line(t: str) -> str:
        return t.split("\n", 1)[0].strip()

    def body_after_heading(t: str) -> str:
        parts = t.split("\n", 1)
        return parts[1].strip() if len(parts) > 1 else ""

    # PASS 1: extract chapters and sections by prefix (order-independent)
    chapter_by_num: dict[str, dict] = {}
    section_by_num: dict[str, dict] = {}
    raw_in_order: list[dict] = []  # raw chunks tagged with classification

    for chunk in raw:
        head = first_line(chunk)
        body = body_after_heading(chunk)

        m_chap = chap_re.match(head)
        m_sect = sect_re.match(head)

        # Chapter heading? Track source position regardless of duplication.
        if m_chap:
            num = m_chap.group(1)
            title_text = m_chap.group(2).strip()
            expected = EXPECTED_CHAPTER_FIRST_WORD.get(num)
            if expected and title_text.startswith(expected):
                if num not in chapter_by_num:
                    chapter_by_num[num] = {
                        "node_id": f"ch-{num}",
                        "title": f"{num}. {title_text}",
                        "num": num,
                        "body": body,
                        "depth": 1,
                        "parent_id": None,
                    }
                # ALWAYS track source position — keeps cur_chapter in pass 2 fresh
                raw_in_order.append({"kind": "chapter", "num": num})
                continue

        # Section heading? Same — track source position regardless of dup.
        if m_sect:
            num = m_sect.group(1)
            chap_num = num.split(".")[0]
            if chap_num in EXPECTED_CHAPTER_FIRST_WORD:
                title_text = m_sect.group(2).strip()
                if num not in section_by_num:
                    section_by_num[num] = {
                        "node_id": f"sec-{num.replace('.', '_')}",
                        "title": f"{num} {title_text}",
                        "num": num,
                        "body": body,
                        "depth": 2,
                        "parent_id": f"ch-{chap_num}",
                    }
                raw_in_order.append({"kind": "section", "num": num})
                continue

        m_rule = rule_re.match(head)
        if m_rule:
            title_text = m_rule.group(1).strip()
            raw_in_order.append({"kind": "rule", "head": head, "body": body, "title": title_text})

    # PASS 2: walk source order, assign each rule to its nearest preceding
    #         section/chapter
    rule_nodes: list[dict] = []
    cur_section_num: str | None = None
    cur_chapter_num: str | None = None
    rule_idx_per_parent: dict[str, int] = {}

    for entry in raw_in_order:
        if entry["kind"] == "chapter":
            cur_chapter_num = entry["num"]
            cur_section_num = None
            continue
        if entry["kind"] == "section":
            cur_section_num = entry["num"]
            cur_chapter_num = entry["num"].split(".")[0]
            continue
        # rule
        if cur_chapter_num is None:
            continue
        parent_id = (
            section_by_num[cur_section_num]["node_id"] if cur_section_num
            else chapter_by_num[cur_chapter_num]["node_id"]
        )
        rule_text = (entry["head"] + "\n\n" + (entry["body"] or "")).strip()
        if len(rule_text) < min_leaf_chars:
            continue
        rule_idx_per_parent[parent_id] = rule_idx_per_parent.get(parent_id, 0) + 1
        rule_idx = rule_idx_per_parent[parent_id]
        # Sub-split big rules
        if len(rule_text) <= max_leaf_chars:
            rule_nodes.append({
                "node_id": f"{parent_id}-r{rule_idx:03d}",
                "title": entry["title"],
                "num": None,
                "body": entry["body"] or "",
                "depth": 3,
                "parent_id": parent_id,
            })
        else:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(max_leaf_chars * 0.9),
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            for sub_i, sub in enumerate(splitter.split_text(rule_text)):
                sub = sub.strip()
                if not sub:
                    continue
                rule_nodes.append({
                    "node_id": f"{parent_id}-r{rule_idx:03d}-{sub_i:02d}",
                    "title": entry["title"],
                    "num": None,
                    "body": sub,
                    "depth": 3,
                    "parent_id": parent_id,
                    "subchunk_index": sub_i,
                })

    # Combine all nodes
    nodes: list[dict] = (
        list(chapter_by_num.values())
        + list(section_by_num.values())
        + rule_nodes
    )

    # PASS 3: build child relationships
    children_by_parent: dict[str, list[dict]] = {}
    for n in nodes:
        if n["parent_id"]:
            children_by_parent.setdefault(n["parent_id"], []).append(n)

    def descendants_of(node_id: str) -> list[dict]:
        out: list[dict] = []
        stack = list(children_by_parent.get(node_id, []))
        while stack:
            child = stack.pop()
            if child["depth"] == 3:
                out.append(child)
            else:
                stack.extend(children_by_parent.get(child["node_id"], []))
        return out

    def breadcrumb_of(node: dict) -> list[str]:
        crumb: list[str] = []
        cur = node
        while cur:
            crumb.insert(0, cur["title"])
            if cur["parent_id"] is None:
                break
            cur = next((x for x in nodes if x["node_id"] == cur["parent_id"]), None)
        return crumb

    # PASS 4: build emitted chunk records
    out_items: list[dict] = []
    for n in nodes:
        crumb = breadcrumb_of(n)
        depth = n["depth"]
        title = n["title"]
        body = n["body"] or ""
        if depth == 3:
            crumb_str = " > ".join(crumb)
            text_for_embed = f"[{crumb_str}]\n\n{title}\n\n{body}".strip()
            desc_ids = ""
        else:
            children = children_by_parent.get(n["node_id"], [])
            child_list = "\n".join(f"- {c['title']}" for c in children) or "(empty)"
            label = "Chapter" if depth == 1 else "Section"
            text_for_embed = (
                f"# {title}\n\n"
                f"{body}\n\n"
                f"This {label.lower()} contains:\n{child_list}"
            ).strip()
            desc_ids = ",".join(d["node_id"] for d in descendants_of(n["node_id"]))

        out_items.append({
            "text": text_for_embed,
            "node_id": n["node_id"],
            "depth": depth,
            "parent_id": n["parent_id"],
            "section_title": title,
            "breadcrumb": " > ".join(crumb),
            "descendant_leaf_ids": desc_ids,
            "n_descendant_leaves": len(desc_ids.split(",")) if desc_ids else 0,
        })
    return out_items


# --- A8: Structure-aware (writing guide) ------------------------------------


def chunk_structure_aware(text: str) -> list[dict]:
    """A8: like C7 but breadcrumb in METADATA, not in text. Each chunk =
    one section's heading + body, with parent chapter/section in metadata."""
    out: list[dict] = []
    for seg in _wg_split_by_heading(text):
        if not seg["body"]:
            continue
        body_text = (
            f"{seg['num']} {seg['title']}\n\n{seg['body']}"
            if seg["num"] else seg["body"]
        )
        breadcrumb: list[str] = []
        if seg.get("parent_chapter"):
            n, t = seg["parent_chapter"]
            breadcrumb.append(f"{n} {t}")
        if seg.get("parent_section"):
            n, t = seg["parent_section"]
            breadcrumb.append(f"{n} {t}")
        if seg["num"]:
            breadcrumb.append(f"{seg['num']} {seg['title']}")
        out.append({
            "text": body_text,
            "section_num": seg["num"],
            "section_title": seg["title"],
            "depth": seg.get("depth", 0),
            "breadcrumb": breadcrumb,
            "parent_chapter_num": seg["parent_chapter"][0] if seg.get("parent_chapter") else None,
            "parent_chapter_title": seg["parent_chapter"][1] if seg.get("parent_chapter") else None,
            "parent_section_num": seg["parent_section"][0] if seg.get("parent_section") else None,
            "parent_section_title": seg["parent_section"][1] if seg.get("parent_section") else None,
        })
    return out


# --- C11: Markdown H2 split (writing guide, pymupdf4llm output) ------------


def chunk_md_h2(
    text: str, min_chars: int = 100, max_chars: int = 2000, overlap_chars: int = 100,
) -> list[dict]:
    """C11: split markdown on every H2 (`## ...`) boundary.

    Used with the pymupdf4llm-extracted writing_guide.md (200+ H2 markers
    representing chapters, sections, and rule-level entries from the WG).

    - Drops chunks below `min_chars` (table column markers like '## **US**').
    - Sub-splits chunks above `max_chars` with `overlap_chars` overlap so
      the alphabetical reference §4.1 doesn't get truncated by the embedder.
    """
    pattern = re.compile(r"^(?=##\s)", re.MULTILINE)
    raw = pattern.split(text)

    items: list[dict] = []
    for raw_chunk in raw:
        raw_chunk = raw_chunk.strip()
        if not raw_chunk:
            continue
        # Extract H2 title for metadata
        m = re.match(r"##\s+\**(?P<title>[^\n*]+?)\**\s*$", raw_chunk.split("\n", 1)[0])
        section_title = m.group("title").strip() if m else None
        # Pre-amble before the first H2 — skip
        if not raw_chunk.startswith("##"):
            if len(raw_chunk) >= min_chars:
                items.append({"text": raw_chunk, "section_title": "Preamble"})
            continue
        if len(raw_chunk) < min_chars:
            continue
        if len(raw_chunk) <= max_chars:
            items.append({"text": raw_chunk, "section_title": section_title})
            continue
        # Sub-split big chunks via the recursive splitter, preserve section title
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=int(max_chars * 0.6),  # tokens, not chars
            chunk_overlap=int(overlap_chars * 0.6),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for i, sub in enumerate(splitter.split_text(raw_chunk)):
            sub = sub.strip()
            if sub:
                items.append({
                    "text": sub,
                    "section_title": section_title,
                    "subchunk_index": i,
                })
    return items


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


def get_chunker(method: str, llm_provider: str = "anthropic",
                llm_model: str | None = None) -> Callable[[str], list]:
    # Simple
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
    if method == "c11": return chunk_md_h2
    # Advanced
    if method == "a1":
        return lambda t: chunk_contextual_retrieval(t, llm_provider=llm_provider, model=llm_model)
    if method == "a2":  return chunk_late
    if method == "a3":
        return lambda t: chunk_propositions(t, llm_provider=llm_provider, model=llm_model)
    if method == "a4":
        return lambda t: chunk_raptor(t, llm_provider=llm_provider, model=llm_model)
    if method == "a5":  return chunk_small_to_big
    if method == "a6":
        return lambda t: chunk_hyde(t, llm_provider=llm_provider, model=llm_model)
    if method == "a7":
        return lambda t: chunk_llm_as_chunker(t, llm_provider=llm_provider, model=llm_model)
    if method == "a8":  return chunk_structure_aware
    if method == "a9":  return chunk_hybrid_small_to_big
    if method == "a10": return chunk_raptor_structural
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


def _stringify_date(v):
    """YAML parses '2020-01-13' as datetime.date — Chroma + JSON need a string.
    Pass strings/None through unchanged."""
    if v is None or isinstance(v, str):
        return v
    try:
        return v.isoformat()  # works for date/datetime
    except AttributeError:
        return str(v)


def _strip_wiki(s):
    """Strip Obsidian-style [[wikilink]] wrappers; pass non-strings through."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("[[") and s.endswith("]]"):
        s = s[2:-2].strip()
    return s


def _first_or_none(seq):
    """First non-empty entry of a list, with [[wiki]] wrappers stripped.
    Accepts a string (treated as a single entry) or a list."""
    if not seq:
        return None
    if isinstance(seq, str):
        return _strip_wiki(seq) or None
    if isinstance(seq, (list, tuple)):
        for item in seq:
            v = _strip_wiki(item)
            if v:
                return v
    return None


def _all_clean(seq):
    """Clean every entry of a list; return list of strings."""
    if not seq:
        return []
    if isinstance(seq, str):
        v = _strip_wiki(seq)
        return [v] if v else []
    if isinstance(seq, (list, tuple)):
        return [v for v in (_strip_wiki(x) for x in seq) if v]
    return []


def make_metadata(
    *,
    article_fm: dict,
    chunker_id: str,
    src_method: str,
    chunk_index: int,
    n_chunks_total: int,
    slug: str,
) -> dict:
    """Build per-chunk metadata.

    Handles two frontmatter dialects we have on different branches:
      Canonical (older): url / date / sector / topic / language / language_original
      Obsidian-style (preproces_docs): source / published / tag (list, [[wiki]]) / lang / author

    Falls back through both. Strips [[...]] wikilink wrappers from tags.
    """
    doc_type = article_fm.get("doc_type", "article")

    tags_clean = _all_clean(article_fm.get("tag") or article_fm.get("tags"))
    primary_tag = tags_clean[0] if tags_clean else None
    secondary_tag = tags_clean[1] if len(tags_clean) >= 2 else None

    return {
        "chunk_id": f"{slug}__{chunker_id}__{chunk_index:04d}",
        "source_slug": slug,
        "source_title": article_fm.get("title"),
        # url canonical → fall back to "source" (Obsidian-style)
        "source_url": (article_fm.get("source_url")
                       or article_fm.get("url")
                       or article_fm.get("source")),
        # date → fall back to "published"
        # YAML parses dates like "2020-01-13" into Python date objects — stringify them.
        "source_date": _stringify_date(article_fm.get("date") or article_fm.get("published")),
        "doc_type": doc_type,
        "channel": "website",
        "content_type": "insight_article" if doc_type == "article" else None,
        # sector: explicit field → first tag (cleaned of [[wiki]] wrappers)
        "sector": article_fm.get("sector") or primary_tag,
        # topic: explicit field → second tag → first tag (different shape)
        "topic": (article_fm.get("topic")
                  or secondary_tag
                  or primary_tag),
        # all tags joined for filtering; primitive types only for Chroma
        "tags": ", ".join(tags_clean) if tags_clean else None,
        "author": _first_or_none(article_fm.get("author")),
        "description": article_fm.get("description"),
        "language_original": (article_fm.get("language_original")
                              or "nl"  # branch's articles are NL→EN translations
                              if (article_fm.get("translated_with")
                                  or src_method in ("gpt-5", "deep-translator",
                                                    "anthropic-claude-sonnet-4-5",
                                                    "anthropic-claude-haiku-4-5",
                                                    "openai-gpt-4o",
                                                    "openai-gpt-4o-mini"))
                              else None),
        # language → fall back to "lang"
        "language_embedded": (article_fm.get("language")
                              or article_fm.get("lang")
                              or "en"),
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
    chunker: Callable[[str], list],
    chunker_id: str,
    src_method: str,
) -> tuple[int, int]:
    """Chunk one article, write JSONL, return (n_chunks, total_chars).
    Chunker output may be list[str] OR list[dict]. Dict items must have a
    'text' key; any other keys merge into the chunk's metadata."""
    fm, body = read_md(src_path)
    items = chunker(body)
    if not items:
        return 0, 0

    slug = src_path.stem
    out_path = out_dir / f"{slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            if isinstance(item, str):
                text = item
                extra: dict = {}
            else:
                text = item.get("text", "")
                extra = {k: v for k, v in item.items() if k != "text"}
            if not text:
                continue
            meta = make_metadata(
                article_fm=fm,
                chunker_id=chunker_id,
                src_method=src_method,
                chunk_index=i,
                n_chunks_total=len(items),
                slug=slug,
            )
            meta.update(extra)
            row = {"id": meta["chunk_id"], "text": text, "metadata": meta}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_chars += len(text)
    return len(items), total_chars


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(method: str, src_method: str, limit: int | None, force: bool,
        llm_provider: str = "anthropic", llm_model: str | None = None) -> None:
    if method not in METHODS:
        raise SystemExit(f"unknown method: {method!r}")
    info = METHODS[method]
    chunker_folder = info["folder"]
    tier = info["tier"]

    # Resolve input source: writing-guide special case vs article translations
    if src_method == "writing-guide":
        if info["applies_to"] != "writing_guide":
            raise SystemExit(
                f"{method} is for {info['applies_to']!r}; use src=deep-translator (or other translation method)"
            )
        wg_path = PROJECT_ROOT / "data" / "writing_guide.md"
        if not wg_path.exists():
            raise SystemExit(f"{wg_path} not found — run scripts/ingest_pdf.py first")
        files = [wg_path]
    else:
        if info["applies_to"] != "article":
            raise SystemExit(
                f"{method} is for {info['applies_to']!r}; use src=writing-guide"
            )
        src_dir = TRANSLATED_PATHS.get(src_method)
        if src_dir is None:
            raise SystemExit(
                f"unknown article source {src_method!r}; "
                f"known: {sorted(TRANSLATED_PATHS)}"
            )
        if not src_dir.exists():
            raise SystemExit(f"source dir not found: {src_dir}")
        files = sorted(src_dir.glob("*.md"))

    out_dir = CHUNKED_BASE / tier / chunker_folder / src_method
    out_dir.mkdir(parents=True, exist_ok=True)

    chunker = get_chunker(method, llm_provider=llm_provider, llm_model=llm_model)
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
    article_fast = [m for m, info in METHODS.items()
                    if info["applies_to"] == "article" and info["scope"] == "fast"]
    article_simple = [m for m, info in METHODS.items()
                      if info["applies_to"] == "article" and info["tier"] == "simple"]
    no_key_methods = [m for m, info in METHODS.items() if info["scope"] in ("fast", "slow")]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        help=f"chunking method: one of {sorted(METHODS)}, "
             f"'all-fast' ({article_fast}), 'all-simple', 'all-no-key' (everything that doesn't need an API key), or 'all-advanced'",
    )
    ap.add_argument("--src", default="deep-translator", help="translation method subfolder, or 'writing-guide'")
    ap.add_argument("--limit", type=int, default=None, help="cap docs for testing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--llm-provider", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--llm-model", default=None, help="override default LLM model")
    args = ap.parse_args()

    if args.method == "all-fast":
        targets = article_fast
    elif args.method == "all-simple":
        targets = [m for m, info in METHODS.items() if info["tier"] == "simple"]
    elif args.method == "all-advanced":
        targets = [m for m, info in METHODS.items() if info["tier"] == "advanced"]
    elif args.method == "all-no-key":
        targets = no_key_methods
    else:
        targets = [args.method]

    for m in targets:
        # auto-pick src for writing-guide-only methods
        info = METHODS[m]
        src = "writing-guide" if info["applies_to"] == "writing_guide" else args.src
        try:
            run(m, src, args.limit, args.force,
                llm_provider=args.llm_provider, llm_model=args.llm_model)
        except SystemExit as e:
            print(f"[chunk] {m}: SKIPPED — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
