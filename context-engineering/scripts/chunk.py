"""Chunk translated articles and the Writing Guide into retrieval-ready pieces.

Strategies:
- fixed_size: 512-token recursive split with 50-token overlap (articles)
- paragraph: split on blank lines (articles)
- heading: split on numbered section headings (writing guide)
"""

from __future__ import annotations

import re
from datetime import date
from typing import Iterable

DATA_DIR = None  # set in main()


def fixed_size_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


def paragraph_chunks(text: str, min_len: int = 50) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > min_len]


def heading_chunks(text: str) -> list[str]:
    pattern = r"(?=\n\d+\.\d+\.?\d*\s)"
    return [c.strip() for c in re.split(pattern, text) if c.strip()]


def make_metadata(
    *,
    chunk_id: str,
    source_title: str,
    source_url: str,
    doc_type: str,
    sector: str | None = None,
    language_original: str | None = None,
    rule_type: str | None = None,
    section_heading: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_title": source_title,
        "source_url": source_url,
        "doc_type": doc_type,
        "channel": "website",
        "content_type": "insight_article",
        "sector": sector,
        "language_original": language_original,
        "language_embedded": "en",
        "status": "approved",
        "date_ingested": date.today().isoformat(),
        "rule_type": rule_type,
        "section_heading": section_heading,
    }


def main() -> None:
    raise NotImplementedError("Wire up after scrape.py and translate.py produce data.")


if __name__ == "__main__":
    main()
