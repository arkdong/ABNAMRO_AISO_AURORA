"""Extract the ABN AMRO Writing Guide PDF into Markdown.

Reads `data/writing_guide.pdf`, strips recurring page footers and TOC pages,
and writes `data/raw/writing_guide.md` with YAML frontmatter and the body
preserved as plain text. Numbered section headings (`1.`, `1.1`, `4.3.1`) are
left in place — chunkers downstream key off them.

Usage:
    python -m scripts.ingest_pdf
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pypdf
import yaml

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "writing_guide.pdf"
OUT_PATH = ROOT / "data" / "raw" / "writing_guide.md"

# Pages 1-4 are cover, intro, and 2-page table of contents.
SKIP_PAGES = {1, 3, 4}

FOOTER_RE = re.compile(r"^\s*\d+\s*\nWriting Guide • [A-Z][a-z]+ \d{4}\s*\n", re.MULTILINE)
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*\nWriting Guide • [A-Z][a-z]+ \d{4}\s*$", re.MULTILINE)


def clean_page(text: str) -> str:
    """Strip the recurring 'N\\nWriting Guide • January 2026' header that
    pypdf places at the top of every extracted page."""
    text = FOOTER_RE.sub("", text, count=1)
    text = PAGE_NUM_RE.sub("", text)
    return text.strip()


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    reader = pypdf.PdfReader(str(PDF_PATH))
    n_pages = len(reader.pages)
    print(f"[ingest] {n_pages} pages in {PDF_PATH.name}", file=sys.stderr)

    page_chunks: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        if i in SKIP_PAGES:
            continue
        text = page.extract_text() or ""
        cleaned = clean_page(text)
        if not cleaned:
            continue
        page_chunks.append(cleaned)

    body = "\n\n".join(page_chunks)
    # Collapse 3+ blank lines and trailing whitespace
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"

    frontmatter = {
        "title": "ABN AMRO Writing Guide",
        "version": "1.1",
        "date": "2026-01",
        "doc_type": "writing_guide",
        "language": "en",
        "source_pdf": "data/writing_guide.pdf",
        "ingested_at": date.today().isoformat(),
    }
    yaml_block = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(f"---\n{yaml_block}\n---\n\n{body}", encoding="utf-8")

    n_chars = len(body)
    print(f"[ingest] wrote {OUT_PATH} ({n_chars:,} chars body, {n_pages - len(SKIP_PAGES)} pages kept)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
