"""Scrape ABN AMRO insight articles from abnamro.nl/zakelijk/insights.

Each article is saved as a JSON file in data/raw/ with metadata:
title, url, date, sector, language ("nl"), body.
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INSIGHTS_INDEX_URL = "https://www.abnamro.nl/nl/zakelijk/insights/"


def discover_article_urls() -> list[str]:
    """Return the list of article URLs to scrape."""
    raise NotImplementedError


def scrape_article(url: str) -> dict:
    """Fetch one article and return a dict with title, date, sector, body, url."""
    raise NotImplementedError


def save(article: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = article["url"].rstrip("/").rsplit("/", 1)[-1]
    path = RAW_DIR / f"{slug}.json"
    path.write_text(json.dumps(article, ensure_ascii=False, indent=2))
    return path


def main() -> None:
    for url in discover_article_urls():
        article = scrape_article(url)
        save(article)


if __name__ == "__main__":
    main()
