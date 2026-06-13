"""Extract the ABN AMRO Writing Guide PDF into Markdown.

Uses **pymupdf4llm** which preserves heading structure (`## ...`), bold
markers (`**...**`) and tables (`| ... |`) — much richer than pypdf's flat
text. Heading-aware chunkers (C7/A8/C11) downstream rely on this.

Reads the shared repo-root Writing Guide PDF, runs the converter, strips repeated footers
("N\\nWriting Guide • January 2026"), filters tiny noise lines (table column
headers like "US"/"UK"), and writes `data/writing_guide.md` with YAML
frontmatter.

Usage:
    python -m scripts.ingest_pdf
    python -m scripts.ingest_pdf --backend pypdf      # legacy fallback
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent       # archive/context-engineering/
ARCHIVE_ROOT = ROOT.parent
PROJECT_ROOT = ARCHIVE_ROOT.parent
PDF_PATH = PROJECT_ROOT / "data" / "Writing Guide 2026-V1.1.pdf"
OUT_PATH = PROJECT_ROOT / "data" / "writing_guide.md"


# ---------------------------------------------------------------------------
# Backend: pymupdf4llm (preferred — preserves headings + tables)
# ---------------------------------------------------------------------------


def extract_pymupdf4llm(pdf_path: Path) -> str:
    import pymupdf4llm

    text = pymupdf4llm.to_markdown(str(pdf_path))
    return _post_process(text)


def _post_process(text: str) -> str:
    """Strip recurring footers, filter noise, normalise whitespace."""
    # Remove footer that appears mid-text: "Writing Guide • January 2026" /
    # "Writing Guide • May 2024", optionally followed by chapter title.
    text = re.sub(
        r"\n+Writing Guide\s*•\s*[A-Z][a-z]+\s*\d{4}\s*\n",
        "\n\n",
        text,
    )
    # Collapse repeated headings of just "**US**" / "**UK**" / "**Noun**"
    # which are table-column markers, not real sections.
    text = re.sub(
        r"^##\s+\*\*(?:US|UK|Noun)\*\*\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Drop "## **Like this**" / "## **Not like this**" example labels —
    # they're inline annotations, not real headings.
    text = re.sub(
        r"^##\s+\*\*(?:Like this|Not like this|Example|Examples?)\*?\*?\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Strip empty headers `## ` or `## **` that may remain
    text = re.sub(r"^##\s*\**\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Backend: pypdf (legacy — fallback for environments without pymupdf4llm)
# ---------------------------------------------------------------------------


SKIP_PAGES_PYPDF = {1, 3, 4}
FOOTER_RE = re.compile(r"^\s*\d+\s*\nWriting Guide • [A-Z][a-z]+ \d{4}\s*\n", re.MULTILINE)
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*\nWriting Guide • [A-Z][a-z]+ \d{4}\s*$", re.MULTILINE)


def extract_pypdf(pdf_path: Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    print(f"[pypdf] {n_pages} pages", file=sys.stderr)
    page_chunks: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        if i in SKIP_PAGES_PYPDF:
            continue
        page_text = page.extract_text() or ""
        page_text = FOOTER_RE.sub("", page_text, count=1)
        page_text = PAGE_NUM_RE.sub("", page_text)
        page_text = page_text.strip()
        if page_text:
            page_chunks.append(page_text)
    body = "\n\n".join(page_chunks)
    return re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("pymupdf4llm", "pypdf"), default="pymupdf4llm",
                    help="extractor to use; pymupdf4llm preserves heading structure")
    args = ap.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    print(f"[ingest] backend={args.backend} src={PDF_PATH.name}", file=sys.stderr)
    if args.backend == "pymupdf4llm":
        body = extract_pymupdf4llm(PDF_PATH)
    else:
        body = extract_pypdf(PDF_PATH)

    # Quick stats for the log
    h2_count = len(re.findall(r"^##\s", body, re.MULTILINE))
    table_rows = len(re.findall(r"^\|", body, re.MULTILINE))

    frontmatter = {
        "title": "ABN AMRO Writing Guide",
        "version": "1.1",
        "date": "2026-01",
        "doc_type": "writing_guide",
        "language": "en",
        "source_pdf": "data/Writing Guide 2026-V1.1.pdf",
        "extractor": args.backend,
        "ingested_at": date.today().isoformat(),
    }
    yaml_block = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(f"---\n{yaml_block}\n---\n\n{body}", encoding="utf-8")
    print(
        f"[ingest] wrote {OUT_PATH} ({len(body):,} chars body, "
        f"H2={h2_count}, table_rows={table_rows})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
