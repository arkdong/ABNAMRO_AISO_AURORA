"""Scrape ABN AMRO insight articles from abnamro.nl/zakelijk/insights.

Discovers article URLs via the public year-by-year sitemap, then fetches each
article and saves a Markdown file (YAML frontmatter + Markdown body) in
data/raw/. Articles older than --since are skipped.

Usage:
    python -m scripts.scrape                       # last 1 year, all articles
    python -m scripts.scrape --since 2025-04-30
    python -m scripts.scrape --limit 20            # cap for testing
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = "https://www.abnamro.nl"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Topic + year combinations exposed by the public sitemap.
# Format: (path, year). Pages older than --since are skipped at runtime.
SITEMAP_PAGES = [
    ("/nl/zakelijk/insights/sitemap/sectoren-en-trends/2024.html", 2024),
    ("/nl/zakelijk/insights/sitemap/sectoren-en-trends/2025.html", 2025),
    ("/nl/zakelijk/insights/sitemap/sectoren-en-trends/2026.html", 2026),
    ("/nl/zakelijk/insights/sitemap/cybersecurity/2023.html", 2023),
    ("/nl/zakelijk/insights/sitemap/cybersecurity/2024.html", 2024),
    ("/nl/zakelijk/insights/sitemap/cybersecurity/2025.html", 2025),
]

DUTCH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "nl,en;q=0.8"})
    return s


def discover_article_urls(s: requests.Session, min_year: int) -> list[str]:
    """Walk sitemap pages whose year >= min_year and collect article URLs."""
    urls: set[str] = set()
    for page, year in SITEMAP_PAGES:
        if year < min_year:
            continue
        try:
            r = s.get(urljoin(ROOT, page), timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[discover] skip {page}: {e}", file=sys.stderr)
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/insights/" not in href or "/sitemap/" in href:
                continue
            if not href.endswith(".html") or "?" in href:
                continue
            full = urljoin(ROOT, href)
            parts = full.split("/insights/", 1)[-1].split("/")
            if len(parts) >= 3 and not parts[-1].startswith("index"):
                urls.add(full)
    return sorted(urls)


def parse_article(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    def og(prop: str) -> str | None:
        tag = soup.find("meta", property=prop)
        return tag.get("content") if tag else None

    title = og("og:title")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None
    canonical = og("og:url") or url

    # Sector + topic from breadcrumb JSON-LD
    sector = None
    topic = None
    for block in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(block.string or "")
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
            for item in data.get("itemListElement", []):
                pos = item.get("position")
                name = item.get("name")
                if pos == 3:
                    topic = name
                elif pos == 4:
                    sector = name

    # Strip noise before grabbing body
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    article_tag = soup.find("article")
    if not article_tag:
        return None

    # Plain-text version (for date detection + length sanity check)
    body_text = article_tag.get_text(separator="\n", strip=True)

    # Markdown body: convert article HTML preserving headings, lists, bold, links
    body_md = markdownify(
        str(article_tag),
        heading_style="ATX",       # use # ## ###
        strip=["img"],             # drop images, keep text only
        bullets="-",
    ).strip()
    # Collapse 3+ blank lines that markdownify sometimes produces
    body_md = re.sub(r"\n{3,}", "\n\n", body_md)

    # Date in body text header (DD/MM/YYYY)
    date_iso = None
    m = DUTCH_DATE_RE.search(body_text)
    if m:
        d, mo, y = m.groups()
        try:
            date_iso = datetime(int(y), int(mo), int(d)).date().isoformat()
        except ValueError:
            date_iso = None

    if not title or len(body_text) < 300:
        return None

    return {
        "title": title,
        "url": canonical,
        "date": date_iso,
        "sector": sector,
        "topic": topic,
        "language": "nl",
        "body": body_md,
    }


def scrape_article(s: requests.Session, url: str) -> dict | None:
    r = s.get(url, timeout=20)
    r.raise_for_status()
    return parse_article(r.text, url)


def save(article: dict) -> Path:
    """Write article as Markdown with YAML frontmatter."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = article["url"].rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
    path = RAW_DIR / f"{slug}.md"
    frontmatter = {
        "title": article["title"],
        "url": article["url"],
        "date": article["date"],
        "sector": article["sector"],
        "topic": article["topic"],
        "language": article["language"],
    }
    yaml_block = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    content = f"---\n{yaml_block}\n---\n\n{article['body']}\n"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    default_since = (date.today() - timedelta(days=365)).isoformat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=default_since, help="ISO date; skip articles older")
    ap.add_argument("--limit", type=int, default=None, help="cap number of articles (testing)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed (only matters if --limit)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if .md already exists")
    args = ap.parse_args()

    cutoff = date.fromisoformat(args.since)
    print(f"[scrape] cutoff date: {cutoff} (articles older than this are skipped)", file=sys.stderr)

    s = session()
    print("[discover] fetching sitemap pages...", file=sys.stderr)
    urls = discover_article_urls(s, min_year=cutoff.year)
    print(f"[discover] found {len(urls)} candidate article URLs", file=sys.stderr)
    if args.limit:
        random.Random(args.seed).shuffle(urls)

    saved = 0
    skipped_old = 0
    skipped_existing = 0
    for url in urls:
        if args.limit and saved >= args.limit:
            break
        slug = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
        out_path = RAW_DIR / f"{slug}.md"
        if out_path.exists() and not args.force:
            skipped_existing += 1
            continue
        try:
            article = scrape_article(s, url)
        except requests.RequestException as e:
            print(f"[scrape] {url} -> error: {e}", file=sys.stderr)
            continue
        if not article:
            print(f"[scrape] {url} -> skipped (no body / title)", file=sys.stderr)
            continue
        if article["date"] and date.fromisoformat(article["date"]) < cutoff:
            skipped_old += 1
            time.sleep(args.delay)
            continue
        path = save(article)
        saved += 1
        print(f"[saved {saved:>3}] {path.name} ({article['sector']}, {article['date']})")
        time.sleep(args.delay)

    print(
        f"[done] saved={saved} skipped_old={skipped_old} skipped_existing={skipped_existing}"
        f" -> {RAW_DIR}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
