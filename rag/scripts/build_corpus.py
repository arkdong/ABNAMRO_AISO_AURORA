"""Concatenate the EN articles into one corpus_en.md for PageIndex markdown indexing.

Each article becomes a level-1 (#) node with its existing ## sections preserved.
Drops YAML frontmatter, image/video/iframe lines, the byline line, and trailing
CTA boilerplate sections. Also emits ``corpus_en_manifest.json`` — one record per
article carrying the frontmatter metadata (title, description, tag, published,
source) for later enrichment of the PageIndex tree.
"""
from pathlib import Path
import json
import re

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTICLES_DIR = REPO_ROOT / "data" / "article" / "en"
OUT_PATH = REPO_ROOT / "rag" / "corpus" / "corpus_en.md"
MANIFEST_PATH = REPO_ROOT / "rag" / "corpus" / "corpus_en_manifest.json"

WIKILINK_RE = re.compile(r"\[\[(.+?)\]\]")

BOILERPLATE_HEADINGS = (
    "## Read more in",
    "## Achter de Cijfers podcast",
    "## In conversation",
    "## More information",
)

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TITLE_RE = re.compile(r'^title:\s*"(.+?)"', re.MULTILINE)
IMAGE_LINE_RE = re.compile(r"^\s*!\[.*?\]\(.*?\).*$", re.MULTILINE)
HTML_TAG_LINE_RE = re.compile(r"^\s*<(iframe|video|source|track|p|a)\b.*$", re.MULTILINE | re.IGNORECASE)
BYLINE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}\s*•.*$", re.MULTILINE)


def extract_title(frontmatter: str, fallback: str) -> str:
    m = TITLE_RE.search(frontmatter)
    return m.group(1).strip() if m else fallback


def strip_frontmatter(text: str) -> tuple[str, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end():]


def _clean_wikilinks(value):
    """Unwrap Obsidian-style ``[[Foo]]`` tags into plain ``Foo`` strings."""
    if isinstance(value, str):
        return WIKILINK_RE.sub(r"\1", value).strip()
    if isinstance(value, list):
        return [_clean_wikilinks(v) for v in value]
    return value


def parse_frontmatter(raw_frontmatter: str) -> dict:
    """Best-effort YAML parse of the frontmatter block. Returns ``{}`` on failure."""
    if not raw_frontmatter:
        return {}
    try:
        data = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def manifest_entry(path: Path, frontmatter: dict, title: str) -> dict:
    """Project frontmatter into the small set of fields the retrieval layer cares about."""
    published = frontmatter.get("published")
    if published is not None:
        published = str(published)
    return {
        "slug": path.stem,
        "title": title,
        "description": _clean_wikilinks(frontmatter.get("description") or "").strip() or None,
        "tags": _clean_wikilinks(frontmatter.get("tag") or []) or [],
        "published": published,
        "source": frontmatter.get("source"),
        "author": _clean_wikilinks(frontmatter.get("author") or []) or [],
    }


def truncate_at_boilerplate(body: str) -> str:
    earliest = len(body)
    for heading in BOILERPLATE_HEADINGS:
        idx = body.find("\n" + heading)
        if idx != -1 and idx < earliest:
            earliest = idx
    return body[:earliest].rstrip()


LEADING_H1_RE = re.compile(r"^\s*#\s+[^\n]+\n+", re.MULTILINE)


def clean_body(body: str) -> str:
    body = IMAGE_LINE_RE.sub("", body)
    body = HTML_TAG_LINE_RE.sub("", body)
    body = BYLINE_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = body.lstrip()
    if body.startswith("# "):
        body = LEADING_H1_RE.sub("", body, count=1)
    return body.strip()


def process_article(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8")
    raw_frontmatter, body = strip_frontmatter(raw)
    fm = parse_frontmatter(raw_frontmatter)
    title = extract_title(raw_frontmatter, fallback=path.stem)
    body = truncate_at_boilerplate(body)
    body = clean_body(body)
    return f"# {title}\n\n{body}\n", manifest_entry(path, fm, title)


def main() -> None:
    article_paths = sorted(ARTICLES_DIR.glob("*.md"))
    print(f"Found {len(article_paths)} articles in {ARTICLES_DIR}")

    sections: list[str] = []
    manifest: list[dict] = []
    for path in article_paths:
        section, entry = process_article(path)
        sections.append(section)
        manifest.append(entry)
    corpus = "\n\n---\n\n".join(sections)

    OUT_PATH.write_text(corpus, encoding="utf-8")
    print(f"Wrote corpus to {OUT_PATH}")
    print(f"Size: {len(corpus):,} chars / {len(corpus.split()):,} words")
    h1_count = corpus.count("\n# ") + (1 if corpus.startswith("# ") else 0)
    h2_count = corpus.count("\n## ")
    print(f"Tree shape preview: {h1_count} H1 (articles) / {h2_count} H2 (sections)")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest to {MANIFEST_PATH}")
    with_desc = sum(1 for e in manifest if e.get("description"))
    with_tags = sum(1 for e in manifest if e.get("tags"))
    print(f"Manifest: {len(manifest)} entries / {with_desc} with description / {with_tags} with tags")


if __name__ == "__main__":
    main()
