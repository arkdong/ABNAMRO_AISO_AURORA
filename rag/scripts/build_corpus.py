"""Concatenate the 10 EN articles into one corpus_en.md for PageIndex markdown indexing.

Each article becomes a level-1 (#) node with its existing ## sections preserved.
Drops YAML frontmatter, image/video/iframe lines, the byline line, and trailing
CTA boilerplate sections.
"""
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTICLES_DIR = REPO_ROOT / "data" / "article" / "en"
OUT_PATH = REPO_ROOT / "rag" / "corpus" / "corpus_en.md"

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


def truncate_at_boilerplate(body: str) -> str:
    earliest = len(body)
    for heading in BOILERPLATE_HEADINGS:
        idx = body.find("\n" + heading)
        if idx != -1 and idx < earliest:
            earliest = idx
    return body[:earliest].rstrip()


def clean_body(body: str) -> str:
    body = IMAGE_LINE_RE.sub("", body)
    body = HTML_TAG_LINE_RE.sub("", body)
    body = BYLINE_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def process_article(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = strip_frontmatter(raw)
    title = extract_title(frontmatter, fallback=path.stem)
    body = truncate_at_boilerplate(body)
    body = clean_body(body)
    return f"# {title}\n\n{body}\n"


def main() -> None:
    article_paths = sorted(ARTICLES_DIR.glob("*.md"))
    print(f"Found {len(article_paths)} articles in {ARTICLES_DIR}")

    sections = [process_article(p) for p in article_paths]
    corpus = "\n\n---\n\n".join(sections)

    OUT_PATH.write_text(corpus, encoding="utf-8")
    print(f"Wrote corpus to {OUT_PATH}")
    print(f"Size: {len(corpus):,} chars / {len(corpus.split()):,} words")
    h1_count = corpus.count("\n# ") + (1 if corpus.startswith("# ") else 0)
    h2_count = corpus.count("\n## ")
    print(f"Tree shape preview: {h1_count} H1 (articles) / {h2_count} H2 (sections)")


if __name__ == "__main__":
    main()
