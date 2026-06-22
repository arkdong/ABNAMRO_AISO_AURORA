"""Build local RAG assets for the standalone AURORA tool server.

The live server intentionally carries copied runtime assets under
``aurora_tool_server/assets/rag``. This script rebuilds those assets from the
repository source documents:

- ``data/article/<lang>/*.md`` for Insights article corpora.
- ``schrijfwijzer.pdf`` for the Dutch writing reference.
- ``Insights_Stijlgids_20250318.pdf`` for the Dutch Insights style guide.
- ``data/insights_stijlgids_en.md`` for the English Insights style guide
  translation.

It emits PageIndex-compatible tree JSON and sparse-vector JSONL chunks. The
vector files are embedding-ready, but also include local term vectors so the
``vector_rag`` backend can work without network-bound embedding generation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ROOT_CORPUS_DIR = REPO_ROOT / "rag" / "corpus"
LIVE_RAG_DIR = REPO_ROOT / "aurora_tool_server" / "assets" / "rag"
SCHRIJFWIJZER_PDF = REPO_ROOT / "schrijfwijzer.pdf"
INSIGHTS_STIJLGIDS_PDF = REPO_ROOT / "Insights_Stijlgids_20250318.pdf"
INSIGHTS_STIJLGIDS_EN_MD = DATA_DIR / "insights_stijlgids_en.md"

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TITLE_RE = re.compile(r'^title:\s*"(.+?)"', re.MULTILINE)
WIKILINK_RE = re.compile(r"\[\[(.+?)\]\]")
IMAGE_LINE_RE = re.compile(r"^\s*!\[.*?\]\(.*?\).*$", re.MULTILINE)
HTML_TAG_LINE_RE = re.compile(
    r"^\s*<(iframe|video|source|track|p|a)\b.*$",
    re.MULTILINE | re.IGNORECASE,
)
BYLINE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}\s*(?:\u2022|•).*$", re.MULTILINE)
LEADING_H1_RE = re.compile(r"^\s*#\s+[^\n]+\n+", re.MULTILINE)
H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
PAGE_SECTION_RE = re.compile(r"^## Page\s+(\d+):\s+(.+?)\s*$", re.MULTILINE)

BOILERPLATE_HEADINGS = (
    "## Read more in",
    "## Achter de Cijfers podcast",
    "## In conversation",
    "## More information",
    "## Lees meer",
    "## Meer informatie",
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "about",
    "what",
    "when",
    "where",
    "which",
    "your",
    "our",
    "are",
    "was",
    "were",
    "een",
    "het",
    "de",
    "en",
    "van",
    "voor",
    "met",
    "dat",
    "die",
    "dit",
    "deze",
    "naar",
    "zijn",
    "wordt",
    "door",
    "als",
    "ook",
    "aan",
    "op",
    "bij",
    "uit",
    "over",
}


def _ensure_dirs() -> None:
    ROOT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_RAG_DIR.mkdir(parents=True, exist_ok=True)


def _clean_wikilinks(value: Any) -> Any:
    if isinstance(value, str):
        return WIKILINK_RE.sub(r"\1", value).strip()
    if isinstance(value, list):
        return [_clean_wikilinks(item) for item in value]
    return value


def _strip_frontmatter(text: str) -> tuple[str, str]:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return "", text
    return match.group(1), text[match.end() :]


def _parse_frontmatter(raw_frontmatter: str) -> dict[str, Any]:
    if not raw_frontmatter:
        return {}
    try:
        data = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_title(raw_frontmatter: str, fallback: str) -> str:
    match = TITLE_RE.search(raw_frontmatter)
    return match.group(1).strip() if match else fallback


def _truncate_at_boilerplate(body: str) -> str:
    earliest = len(body)
    for heading in BOILERPLATE_HEADINGS:
        index = body.find("\n" + heading)
        if index != -1 and index < earliest:
            earliest = index
    return body[:earliest].rstrip()


def _clean_article_body(body: str) -> str:
    body = IMAGE_LINE_RE.sub("", body)
    body = HTML_TAG_LINE_RE.sub("", body)
    body = BYLINE_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body).lstrip()
    if body.startswith("# "):
        body = LEADING_H1_RE.sub("", body, count=1)
    return body.strip()


def _manifest_entry(path: Path, frontmatter: dict[str, Any], title: str, lang: str) -> dict[str, Any]:
    published = frontmatter.get("published")
    if published is not None:
        published = str(published)
    return {
        "slug": path.stem,
        "title": title,
        "description": _clean_wikilinks(frontmatter.get("description") or "") or None,
        "tags": _clean_wikilinks(frontmatter.get("tag") or []) or [],
        "published": published,
        "source": frontmatter.get("source"),
        "author": _clean_wikilinks(frontmatter.get("author") or []) or [],
        "lang": lang,
    }


def _summary(text: str, max_chars: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _split_h2_sections(body: str) -> list[tuple[str, str, int]]:
    matches = list(H2_RE.finditer(body))
    sections: list[tuple[str, str, int]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        line_num = body[:start].count("\n") + 1
        title = match.group(1).strip()
        sections.append((title, body[start:end].strip(), line_num))
    return sections


def _article_tree_node(path: Path, index: int, lang: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    raw_frontmatter, body = _strip_frontmatter(raw)
    frontmatter = _parse_frontmatter(raw_frontmatter)
    title = _extract_title(raw_frontmatter, fallback=path.stem)
    body = _clean_article_body(_truncate_at_boilerplate(body))
    article_text = f"# {title}\n\n{body}\n"
    entry = _manifest_entry(path, frontmatter, title, lang)

    children: list[dict[str, Any]] = []
    for child_index, (section_title, section_text, line_num) in enumerate(
        _split_h2_sections(body),
        start=1,
    ):
        children.append(
            {
                "title": section_title,
                "node_id": f"{index:04d}.{child_index:04d}",
                "line_num": line_num,
                "text": section_text,
                "summary": _summary(section_text),
            }
        )

    node = {
        "title": title,
        "node_id": f"{index:04d}",
        "line_num": 1,
        "text": article_text,
        "nodes": children,
        "prefix_summary": entry.get("description") or _summary(body),
        "tags": entry.get("tags") or [],
        "published": entry.get("published"),
        "source": entry.get("source"),
        "slug": entry.get("slug"),
        "lang": lang,
    }
    return article_text, node, entry


def build_article_corpus(lang: str) -> dict[str, Any]:
    articles_dir = DATA_DIR / "article" / lang
    if not articles_dir.is_dir():
        raise FileNotFoundError(f"Missing article directory: {articles_dir}")

    article_paths = sorted(path for path in articles_dir.glob("*.md") if not path.name.startswith("."))
    sections: list[str] = []
    nodes: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, path in enumerate(article_paths):
        article_text, node, entry = _article_tree_node(path, index, lang)
        sections.append(article_text)
        nodes.append(node)
        manifest.append(entry)

    corpus_text = "\n\n---\n\n".join(sections)
    structure = {
        "doc_name": f"corpus_{lang}",
        "line_count": len(corpus_text.splitlines()),
        "structure": nodes,
    }

    (ROOT_CORPUS_DIR / f"corpus_{lang}.md").write_text(corpus_text, encoding="utf-8")
    (ROOT_CORPUS_DIR / f"corpus_{lang}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (ROOT_CORPUS_DIR / f"corpus_{lang}_structure.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (LIVE_RAG_DIR / f"corpus_{lang}_structure.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return structure


def _extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _clean_pdf_page(page_text: str) -> str:
    lines: list[str] = []
    for line in page_text.splitlines():
        line = line.replace("Schrijfwijzer • Nov 2025", "")
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.isdigit():
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _page_title(page_text: str, page_num: int) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    preferred = (
        "Even vooraf",
        "Inhoud",
        "Voorbereiden",
        "Structuur",
        "Formuleren",
        "Correctheid",
        "Inclusief schrijven",
        "Index",
    )
    for line in lines[:12]:
        if line in preferred or re.match(r"^\d+(?:\.\d+)*\.?\s+\S", line):
            return line
    return lines[0] if lines else f"Pagina {page_num}"


def build_schrijfwijzer_tree(pdf_path: Path = SCHRIJFWIJZER_PDF) -> list[dict[str, Any]]:
    raw_text = _extract_pdf_text(pdf_path)
    pages = [_clean_pdf_page(page) for page in raw_text.split("\f")]
    pages = [page for page in pages if page.strip()]
    children: list[dict[str, Any]] = []
    combined_parts: list[str] = []
    for page_num, page in enumerate(pages, start=1):
        title = _page_title(page, page_num)
        text = f"## {title}\n\n{page}"
        combined_parts.append(text)
        children.append(
            {
                "title": title,
                "node_id": f"schrijfwijzer.{page_num:04d}",
                "line_num": page_num,
                "page_index": page_num,
                "text": text,
                "summary": _summary(page),
            }
        )

    root_text = "# Schrijfwijzer\n\n" + "\n\n".join(combined_parts)
    tree = [
        {
            "title": "Schrijfwijzer",
            "node_id": "schrijfwijzer",
            "line_num": 1,
            "text": root_text,
            "nodes": children,
            "summary": (
                "Nederlandse ABN AMRO Schrijfwijzer met afspraken, handvatten "
                "en tips voor duidelijke, inclusieve en consistente teksten."
            ),
            "source": str(pdf_path.relative_to(REPO_ROOT)),
            "lang": "nl",
        }
    ]
    (ROOT_CORPUS_DIR / "schrijfwijzer_tree.json").write_text(
        json.dumps(tree, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (LIVE_RAG_DIR / "schrijfwijzer_tree.json").write_text(
        json.dumps(tree, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return tree


def _insights_page_title(page_text: str, page_num: int) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines[:10]:
        if re.fullmatch(r"\d+(?:\s+\d+)*", line):
            continue
        return line
    return lines[0] if lines else f"Pagina {page_num}"


def _clean_insights_pdf_page(page_text: str) -> str:
    lines: list[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Slide exports include small positioning markers as isolated numbers.
        if re.fullmatch(r"\d+(?:\s+\d+)*", stripped):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _build_page_tree(
    *,
    title: str,
    node_id: str,
    pages: list[tuple[int, str, str]],
    summary: str,
    source: str,
    lang: str,
) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    combined_parts: list[str] = []
    for page_num, page_title, page_body in pages:
        page_body = page_body.strip()
        text = f"## {page_title}\n\n{page_body}" if page_body else f"## {page_title}"
        combined_parts.append(text)
        children.append(
            {
                "title": page_title,
                "node_id": f"{node_id}.{page_num:04d}",
                "line_num": page_num,
                "page_index": page_num,
                "text": text,
                "summary": _summary(page_body),
                "source": source,
                "lang": lang,
            }
        )

    root_text = f"# {title}\n\n" + "\n\n".join(combined_parts)
    return [
        {
            "title": title,
            "node_id": node_id,
            "line_num": 1,
            "text": root_text,
            "nodes": children,
            "summary": summary,
            "source": source,
            "lang": lang,
        }
    ]


def _write_tree_asset(filename: str, tree: list[dict[str, Any]]) -> None:
    payload = json.dumps(tree, indent=2, ensure_ascii=False) + "\n"
    (ROOT_CORPUS_DIR / filename).write_text(payload, encoding="utf-8")
    (LIVE_RAG_DIR / filename).write_text(payload, encoding="utf-8")


def build_insights_stijlgids_nl_tree(
    pdf_path: Path = INSIGHTS_STIJLGIDS_PDF,
) -> list[dict[str, Any]]:
    raw_text = _extract_pdf_text(pdf_path)
    raw_pages = [_clean_insights_pdf_page(page) for page in raw_text.split("\f")]
    pages: list[tuple[int, str, str]] = []
    for page_num, page in enumerate(raw_pages, start=1):
        if not page.strip():
            continue
        pages.append((page_num, _insights_page_title(page, page_num), page))

    tree = _build_page_tree(
        title="Stijlgids voor Insights",
        node_id="insights_stijlgids_nl",
        pages=pages,
        summary=(
            "Nederlandse Insights-stijlgids voor formats, SEO, Schrijfwijzer, "
            "Brand Experience, toegankelijkheid, beeld, video, illustraties, "
            "grafieken en pdf-documenten."
        ),
        source=str(pdf_path.relative_to(REPO_ROOT)),
        lang="nl",
    )
    _write_tree_asset("insights_stijlgids_nl_tree.json", tree)
    return tree


def _parse_translated_insights_pages(markdown_path: Path) -> list[tuple[int, str, str]]:
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Missing translated Insights style guide: {markdown_path}")
    raw = markdown_path.read_text(encoding="utf-8")
    _, body = _strip_frontmatter(raw)
    matches = list(PAGE_SECTION_RE.finditer(body))
    if not matches:
        raise ValueError(f"No translated page sections found in {markdown_path}")

    pages: list[tuple[int, str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        page_num = int(match.group(1))
        title = match.group(2).strip()
        page_body = body[start:end].strip()
        pages.append((page_num, title, page_body))
    return pages


def build_insights_stijlgids_en_tree(
    markdown_path: Path = INSIGHTS_STIJLGIDS_EN_MD,
) -> list[dict[str, Any]]:
    pages = _parse_translated_insights_pages(markdown_path)
    tree = _build_page_tree(
        title="Insights Style Guide",
        node_id="insights_stijlgids_en",
        pages=pages,
        summary=(
            "English translation of the Insights style guide covering formats, "
            "SEO, the Writing Guide, Brand Experience, accessibility, imagery, "
            "video, illustrations, charts and PDF documents."
        ),
        source=str(markdown_path.relative_to(REPO_ROOT)),
        lang="en",
    )
    _write_tree_asset("insights_stijlgids_en_tree.json", tree)
    return tree


def build_insights_stijlgids_trees() -> None:
    build_insights_stijlgids_nl_tree()
    build_insights_stijlgids_en_tree()


def _walk(nodes: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for node in nodes:
        yield node
        yield from _walk(node.get("nodes") or [])


def _tokens(text: str, *, min_len: int = 3) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) >= min_len and token.lower() not in STOPWORDS
    ]


def _top_terms(text: str, *, limit: int = 96) -> dict[str, float]:
    counts = Counter(_tokens(text))
    if not counts:
        return {}
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    most_common = counts.most_common(limit)
    return {term: round(count / norm, 6) for term, count in most_common}


def _chunk_words(text: str, *, max_words: int = 260, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks: list[str] = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + max_words]
        if len(chunk) < 40 and chunks:
            break
        chunks.append(" ".join(chunk))
    return chunks


def _vector_records_from_tree(
    *,
    source_doc: str,
    nodes: Iterable[dict[str, Any]],
    language: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in _walk(nodes):
        text = str(node.get("text") or "").strip()
        if not text:
            continue
        title = str(node.get("title") or "")
        for chunk_index, chunk in enumerate(_chunk_words(text)):
            record_id = f"{source_doc}:{node.get('node_id')}:{chunk_index:03d}"
            records.append(
                {
                    "id": record_id,
                    "source_doc": source_doc,
                    "node_id": str(node.get("node_id") or ""),
                    "chunk_index": chunk_index,
                    "title": title,
                    "content": chunk,
                    "language": language,
                    "line_num": node.get("line_num") or node.get("page_index"),
                    "source_url": node.get("source"),
                    "tags": node.get("tags") or [],
                    "published": node.get("published"),
                    "terms": _top_terms(" ".join([title, chunk, " ".join(node.get("tags") or [])])),
                }
            )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _load_tree_asset(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return str(data.get("doc_name") or path.stem.removesuffix("_structure")), list(data.get("structure") or [])
    return path.stem.removesuffix("_tree"), list(data)


def build_vector_assets() -> None:
    assets = [
        ("corpus_en", LIVE_RAG_DIR / "corpus_en_structure.json", "en"),
        ("corpus_nl", LIVE_RAG_DIR / "corpus_nl_structure.json", "nl"),
        ("writing_guide", LIVE_RAG_DIR / "writing_guide_tree.json", "en"),
        ("schrijfwijzer", LIVE_RAG_DIR / "schrijfwijzer_tree.json", "nl"),
        ("insights_stijlgids_en", LIVE_RAG_DIR / "insights_stijlgids_en_tree.json", "en"),
        ("insights_stijlgids_nl", LIVE_RAG_DIR / "insights_stijlgids_nl_tree.json", "nl"),
    ]
    for source_doc, path, language in assets:
        if not path.is_file():
            continue
        _, nodes = _load_tree_asset(path)
        records = _vector_records_from_tree(
            source_doc=source_doc,
            nodes=nodes,
            language=language,
        )
        _write_jsonl(LIVE_RAG_DIR / f"vector_{source_doc}.jsonl", records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AURORA RAG assets")
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip rebuilding the Dutch Schrijfwijzer PDF tree.",
    )
    parser.add_argument(
        "--skip-insights-style-guide",
        action="store_true",
        help="Skip rebuilding the Insights style guide trees.",
    )
    args = parser.parse_args()

    _ensure_dirs()
    build_article_corpus("nl")
    if not args.skip_pdf:
        build_schrijfwijzer_tree()
    if not args.skip_insights_style_guide:
        build_insights_stijlgids_trees()
    build_vector_assets()
    print(f"Wrote PageIndex and vector assets to {LIVE_RAG_DIR}")


if __name__ == "__main__":
    main()
