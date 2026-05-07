"""Single-article scraper for ABN AMRO Insights.

Fetches one article URL, parses with BeautifulSoup, evaluates the Obsidian
Web Clipper template (mirrored from `examples/obsidian_template.json`), and
writes a Markdown file with YAML frontmatter. Filename is the article title.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from .template_engine import evaluate

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Mirror of examples/obsidian_template.json — kept inline so the scraper has
# zero external file dependency at runtime.
PROPERTIES: list[dict[str, str]] = [
    {"name": "title", "value": "{{title}}", "type": "text"},
    {"name": "source", "value": "{{url}}", "type": "text"},
    {"name": "lang", "value": "{{selector:html?lang}}", "type": "text"},
    {
        "name": "published",
        "value": (
            '{{selector:[data-component-type="news-article-intro"] '
            ".news-article-intro-meta-data p"
            '|split:"•"|first|trim|date:("YYYY-MM-DD","DD/MM/YYYY")}}'
        ),
        "type": "date",
    },
    {
        "name": "author",
        "value": (
            "{{selector:[data-component-type=news-article-intro] "
            "a.author-snippet-component .title-wrapper > p"
            '|wikilink|join:", "}}'
        ),
        "type": "multitext",
    },
    {"name": "description", "value": "{{description}}", "type": "text"},
    {
        "name": "tag",
        "value": (
            "{{selector:#bottomChips [data-component-type=chip] a.anchor-chip"
            '|wikilink|join:", "}}'
        ),
        "type": "multitext",
    },
]


# ---- HTTP -------------------------------------------------------------------


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "nl,en;q=0.8"})
    return s


# ---- frontmatter rendering --------------------------------------------------
# Hand-rolled YAML emitter that matches data/article/<lang>/ formatting:
#   - mapping keys are bare
#   - string values are double-quoted
#   - dates are emitted as bare YYYY-MM-DD
#   - list items live two spaces under their key
# Using PyYAML's representers instead got us quoted keys, which doesn't match.


def _yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_frontmatter(fm: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in fm.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(
                    f"  - {_yaml_quote(item) if isinstance(item, str) else item}"
                )
        elif isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            lines.append(f"{key}: {value.isoformat()}")
        elif isinstance(value, str):
            lines.append(f"{key}: {_yaml_quote(value)}")
        elif value is None:
            lines.append(f"{key}: ")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ---- parsing ----------------------------------------------------------------


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r'[/\\:*?"<>|\n\r\t]+', " ", title).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "untitled"


def _extract_body_md(soup: BeautifulSoup) -> str:
    article = soup.find("article")
    if not article:
        return ""
    for tag in article(["script", "style", "noscript"]):
        tag.decompose()
    body = markdownify(str(article), heading_style="ATX", bullets="-").strip()
    return re.sub(r"\n{3,}", "\n\n", body)


def parse_html(html: str, url: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")
    fm: dict[str, Any] = {}
    for prop in PROPERTIES:
        is_list = prop["type"] == "multitext"
        fm[prop["name"]] = evaluate(prop["value"], soup, url, is_list=is_list)
    body = _extract_body_md(soup)
    if not fm.get("title") or len(body) < 200:
        return None
    return {"frontmatter": fm, "body": body}


def scrape_url(url: str, sess: requests.Session | None = None) -> dict[str, Any] | None:
    s = sess or session()
    r = s.get(url, timeout=20)
    r.raise_for_status()
    return parse_html(r.text, url)


# ---- output -----------------------------------------------------------------


def save_article(article: dict[str, Any], out_dir: Path, force: bool = False) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fm = dict(article["frontmatter"])

    # Emit `published` as a real date so it serialises unquoted.
    pub = fm.get("published")
    if isinstance(pub, str) and pub:
        try:
            fm["published"] = datetime.date.fromisoformat(pub)
        except ValueError:
            pass

    filename = _safe_filename(fm["title"]) + ".md"
    path = out_dir / filename
    if path.exists() and not force:
        return None

    yaml_block = _render_frontmatter(fm)
    path.write_text(f"---\n{yaml_block}\n---\n{article['body']}\n", encoding="utf-8")
    return path
