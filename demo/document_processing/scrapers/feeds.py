"""Feed crawlers for ABN AMRO Insights.

Two discovery modes:

- `crawl_actueel(url)`         — static HTML link extraction. Works when the
                                  listing page is server-rendered. Returns 0 if
                                  the page is JS-rendered (currently the case
                                  for `…/sectoren-en-trends/actueel.html`).
- `crawl_sitemap_xml(url)`     — parses an XML sitemap (e.g.
                                  `…/zakelijk/sitemap.xml`) which is fully
                                  populated and is the reliable discovery
                                  source. ~3000 URLs project-wide; filter with
                                  `--path-contains` or `--sector` downstream.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .abnamro_scraper import session

ROOT = "https://www.abnamro.nl"
DEFAULT_FEED = f"{ROOT}/nl/zakelijk/insights/sectoren-en-trends/actueel.html"
DEFAULT_SITEMAP_XML = f"{ROOT}/nl/zakelijk/sitemap.xml"


def _looks_like_article(href: str) -> bool:
    if "/insights/" not in href:
        return False
    if "/sitemap/" in href or "/actueel" in href:
        return False
    if not href.endswith(".html") or "?" in href:
        return False
    parts = href.split("/insights/", 1)[-1].rstrip("/").split("/")
    if len(parts) < 3:
        return False
    if parts[-1].startswith("index"):
        return False
    return True


def crawl_actueel(
    url: str = DEFAULT_FEED,
    sess: requests.Session | None = None,
) -> list[str]:
    """Return a sorted list of article URLs linked from a feed page."""
    s = sess or session()
    r = s.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(ROOT, a["href"])
        if _looks_like_article(full):
            found.add(full)
    return sorted(found)


_LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


def crawl_sitemap_xml(
    url: str = DEFAULT_SITEMAP_XML,
    path_contains: str | None = None,
    sess: requests.Session | None = None,
) -> list[str]:
    """Return article URLs from an XML sitemap, optionally filtered by path.

    `path_contains` is a substring matched against each `<loc>` URL — e.g.
    "/sectoren-en-trends/technologie/" to keep only TMT-sector articles.
    Cheap pre-filter that avoids fetching every article just to discard most.
    """
    s = sess or session()
    r = s.get(url, timeout=60)
    r.raise_for_status()
    locs = _LOC_RE.findall(r.text)
    out: set[str] = set()
    for loc in locs:
        if not _looks_like_article(loc):
            continue
        if path_contains and path_contains not in loc:
            continue
        out.add(loc)
    return sorted(out)
