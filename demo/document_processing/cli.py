"""CLI for the document_processing module (scrape + translate).

Usage:
    python -m document_processing scrape-url <article-url>
    python -m document_processing scrape-feed [--sitemap-xml] [--path-contains ...] [--sector ...]
    python -m document_processing translate [--src dir] [--dst dir] [--file path] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

from . import markdown_io
from .scrapers.abnamro_scraper import save_article, scrape_url, session
from .scrapers.feeds import (
    DEFAULT_FEED,
    DEFAULT_SITEMAP_XML,
    crawl_actueel,
    crawl_sitemap_xml,
)

DEFAULT_OUT_DIR = Path("data/article/nl")
DEFAULT_TRANSLATE_SRC = Path("data/article/nl")
DEFAULT_TRANSLATE_DST = Path("data/article/en")


def _matches_sector(article: dict, sector: str | None) -> bool:
    """True if the article's `tag` list contains the requested sector.

    Tags arrive as wikilinks (e.g. "[[Technologie, Media & Telecom]]"); we
    strip the brackets and do case-insensitive equality.
    """
    if not sector:
        return True
    target = sector.strip().lower()
    for t in article["frontmatter"].get("tag") or []:
        if t.strip("[] ").lower() == target:
            return True
    return False


def cmd_url(args: argparse.Namespace) -> int:
    article = scrape_url(args.url)
    if not article:
        print(f"[error] could not parse {args.url}", file=sys.stderr)
        return 1
    out = save_article(article, Path(args.out_dir), force=args.force)
    if out is None:
        print(f"[skip] already exists in {args.out_dir} (use --force to overwrite)")
        return 0
    print(f"[saved] {out}")
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    sess = session()
    if args.sitemap_xml:
        print(f"[discover] reading sitemap {args.sitemap_xml}", file=sys.stderr)
        urls = crawl_sitemap_xml(
            args.sitemap_xml, path_contains=args.path_contains, sess=sess
        )
    else:
        print(f"[discover] fetching {args.url}", file=sys.stderr)
        urls = crawl_actueel(args.url, sess=sess)
    print(f"[discover] found {len(urls)} article URL(s)", file=sys.stderr)
    if args.limit:
        urls = urls[: args.limit]

    saved = skipped = errored = 0
    for u in urls:
        try:
            article = scrape_url(u, sess=sess)
        except requests.RequestException as e:
            print(f"[error] {u} -> {e}", file=sys.stderr)
            errored += 1
            continue
        if not article:
            print(f"[skip] {u} -> empty body", file=sys.stderr)
            skipped += 1
            continue
        if not _matches_sector(article, args.sector):
            skipped += 1
            time.sleep(args.delay)
            continue
        out = save_article(article, Path(args.out_dir), force=args.force)
        if out is None:
            skipped += 1
        else:
            saved += 1
            print(f"[saved {saved:>3}] {out.name}")
        time.sleep(args.delay)

    print(
        f"[done] saved={saved} skipped={skipped} errored={errored} -> {args.out_dir}",
        file=sys.stderr,
    )
    return 0


def _translate_one(
    translator,
    src_path: Path,
    dst_dir: Path,
    force: bool,
) -> tuple[str, dict | None, float]:
    """Translate one NL file and write EN. Returns (status, usage, elapsed_s)."""
    dst_path = dst_dir / src_path.name  # filename = NL title; alignment key
    if dst_path.exists() and not force:
        return ("skipped", None, 0.0)

    fm_nl, body_nl = markdown_io.parse_file(src_path)

    title_nl = str(fm_nl.get("title", ""))
    description_nl = str(fm_nl.get("description", ""))

    print(
        f"[call ] {src_path.name}  ({len(body_nl):,} chars) → OpenAI...",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    result = translator.translate(title_nl, description_nl, body_nl)
    elapsed = time.time() - t0

    # Build EN frontmatter: keep ordering, swap lang, replace title/description.
    fm_en: dict = {}
    for key, value in fm_nl.items():
        if key == "title":
            fm_en["title"] = result.title_en
        elif key == "lang":
            fm_en["lang"] = "en"
        elif key == "description":
            fm_en["description"] = result.description_en
        else:
            fm_en[key] = value

    # Optional structural sanity check; tag the file if it fails.
    from .translators.openai_translator import check_structure

    check = check_structure(body_nl, result.body_en)
    if not check.ok:
        fm_en["review_needed"] = True
        fm_en["review_issues"] = check.issues

    markdown_io.write_file(dst_path, fm_en, result.body_en)
    return ("saved", translator.last_usage, elapsed)


def cmd_translate(args: argparse.Namespace) -> int:
    # Load .env if present so users don't need to export the key manually.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Lazy import so scrape commands don't pay openai's import cost.
    from .translators import (
        FALLBACK_KEY_ENV,
        OpenAITranslator,
        TRANSLATION_KEY_ENV,
        resolve_api_key,
    )

    if not resolve_api_key():
        print(
            f"[error] No API key found. Set {TRANSLATION_KEY_ENV} (preferred, "
            f"so translation spend is tracked separately) or {FALLBACK_KEY_ENV} "
            "in your environment or .env file.",
            file=sys.stderr,
        )
        return 2

    using_dedicated = bool(os.environ.get(TRANSLATION_KEY_ENV))
    print(
        f"[auth] using {TRANSLATION_KEY_ENV if using_dedicated else FALLBACK_KEY_ENV}",
        file=sys.stderr,
    )

    translator = OpenAITranslator(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
    )

    if args.file:
        files = [Path(args.file)]
    else:
        src = Path(args.src)
        files = sorted(src.glob("*.md"))
    if args.limit:
        files = files[: args.limit]

    dst_dir = Path(args.dst)
    print(
        f"[translate] model={args.model}  src={args.src or args.file}  dst={dst_dir}  "
        f"count={len(files)}",
        file=sys.stderr,
    )

    saved = skipped = errored = 0
    total_in = total_out = total_cached = 0

    for f in files:
        try:
            status, usage, elapsed = _translate_one(translator, f, dst_dir, args.force)
        except Exception as e:
            print(f"[error] {f.name} -> {e}", file=sys.stderr)
            errored += 1
            continue

        if status == "skipped":
            skipped += 1
            print(f"[skip ] {f.name}  (already exists; --force to overwrite)")
        else:
            saved += 1
            if usage:
                total_in += usage.get("input", 0)
                total_out += usage.get("output", 0)
                total_cached += usage.get("cached", 0)
            tok_str = (
                f" [in={usage['input']:>5d} out={usage['output']:>5d} cached={usage['cached']:>5d} {elapsed:5.1f}s]"
                if usage
                else f" [{elapsed:.1f}s]"
            )
            print(f"[saved {saved:>3}] {f.name}{tok_str}")
        time.sleep(args.delay)

    # gpt-5 list price as of Aug 2025 launch — adjust if model changes.
    PRICING = {
        "gpt-5":      {"in": 1.25, "out": 10.00, "cached_in": 0.125},
        "gpt-5-mini": {"in": 0.25, "out":  2.00, "cached_in": 0.025},
        "gpt-5-nano": {"in": 0.05, "out":  0.40, "cached_in": 0.005},
    }
    px = PRICING.get(args.model)
    cost_str = ""
    if px and saved:
        non_cached_in = max(total_in - total_cached, 0)
        cost = (
            non_cached_in / 1_000_000 * px["in"]
            + total_cached  / 1_000_000 * px["cached_in"]
            + total_out     / 1_000_000 * px["out"]
        )
        cost_str = f"  est_cost=${cost:.4f}"

    print(
        f"[done] saved={saved} skipped={skipped} errored={errored}  "
        f"tokens in={total_in:,} out={total_out:,} cached={total_cached:,}{cost_str}",
        file=sys.stderr,
    )
    return 0 if errored == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(prog="document_processing")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_url = sub.add_parser("scrape-url", help="Scrape a single article URL.")
    p_url.add_argument("url")
    p_url.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p_url.add_argument("--force", action="store_true")
    p_url.set_defaults(func=cmd_url)

    p_feed = sub.add_parser(
        "scrape-feed",
        help="Crawl a feed page (default: actueel.html) and scrape every article.",
    )
    p_feed.add_argument("--url", default=DEFAULT_FEED, help="Feed page (used unless --sitemap-xml is set).")
    p_feed.add_argument(
        "--sitemap-xml",
        nargs="?",
        const=DEFAULT_SITEMAP_XML,
        default=None,
        help=(
            "Discover URLs from an XML sitemap instead of a feed page. "
            "Reliable when the feed is JS-rendered. Pass without value to use "
            f"the default ({DEFAULT_SITEMAP_XML})."
        ),
    )
    p_feed.add_argument(
        "--path-contains",
        default=None,
        help='Sitemap pre-filter: keep only URLs whose path contains this substring '
             '(e.g. "/sectoren-en-trends/technologie/").',
    )
    p_feed.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p_feed.add_argument("--limit", type=int, default=None)
    p_feed.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    p_feed.add_argument("--force", action="store_true")
    p_feed.add_argument(
        "--sector",
        default=None,
        help='Post-parse filter: only save articles whose `tag` list contains this sector '
             '(e.g. "Technologie, Media & Telecom").',
    )
    p_feed.set_defaults(func=cmd_feed)

    p_tr = sub.add_parser(
        "translate",
        help="Translate NL articles in --src to EN files in --dst using OpenAI.",
    )
    p_tr.add_argument("--src", default=str(DEFAULT_TRANSLATE_SRC))
    p_tr.add_argument("--dst", default=str(DEFAULT_TRANSLATE_DST))
    p_tr.add_argument("--file", default=None, help="Translate just this single file.")
    p_tr.add_argument("--model", default="gpt-5", help="OpenAI model id (default: gpt-5).")
    p_tr.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="minimal",
        help="GPT-5 reasoning effort. Translation only needs 'minimal'; "
             "anything higher just burns tokens and stalls the response.",
    )
    p_tr.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="seconds before an in-flight call is aborted (default: 180)",
    )
    p_tr.add_argument("--limit", type=int, default=None)
    p_tr.add_argument("--delay", type=float, default=0.5, help="seconds between calls")
    p_tr.add_argument("--force", action="store_true", help="overwrite existing EN files")
    p_tr.set_defaults(func=cmd_translate)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
