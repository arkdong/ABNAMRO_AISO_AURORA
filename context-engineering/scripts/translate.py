"""Translate Dutch article bodies to English with deep-translator.

Google Translate has a ~5000-char limit per request, so longer articles are
split on paragraph boundaries, translated in batches, and re-stitched.

Reads data/raw/*.json, writes data/translated/*.json, preserving all metadata
and adding language_original="nl", language_embedded="en".

Skips files that already exist in data/translated/ unless --force is passed.

Usage:
    python -m scripts.translate
    python -m scripts.translate --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "translated"

# Google Translate's hard limit is 5000 chars; pad below for safety.
MAX_CHARS = 4500


def batch_paragraphs(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into batches of paragraphs each <= max_chars."""
    paragraphs = text.split("\n\n")
    batches: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        # If a single paragraph exceeds the limit, hard-split on newlines.
        if len(p) > max_chars:
            for line in p.split("\n"):
                if cur_len + len(line) + 1 > max_chars and cur:
                    batches.append("\n".join(cur))
                    cur, cur_len = [], 0
                cur.append(line)
                cur_len += len(line) + 1
            continue
        if cur_len + len(p) + 2 > max_chars and cur:
            batches.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        batches.append("\n\n".join(cur))
    return batches


def translate_text(text: str, translator: GoogleTranslator) -> str:
    batches = batch_paragraphs(text)
    out: list[str] = []
    for b in batches:
        if not b.strip():
            continue
        out.append(translator.translate(b))
    return "\n\n".join(out)


def translate_article(article: dict, translator: GoogleTranslator) -> dict:
    title_en = translator.translate(article["title"]) if article.get("title") else None
    body_en = translate_text(article["body"], translator)
    return {
        **article,
        "title": title_en or article.get("title"),
        "title_original": article.get("title"),
        "body": body_en,
        "body_original": article["body"],
        "language_original": "nl",
        "language_embedded": "en",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-translate already-translated files")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between articles")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    translator = GoogleTranslator(source="nl", target="en")

    raw_files = sorted(RAW_DIR.glob("*.json"))
    print(f"[translate] {len(raw_files)} raw files found", file=sys.stderr)

    done = 0
    for src in raw_files:
        out_path = OUT_DIR / src.name
        if out_path.exists() and not args.force:
            print(f"[skip ] {src.name} (already translated)")
            continue
        article = json.loads(src.read_text())
        n_chars = len(article.get("body", ""))
        try:
            translated = translate_article(article, translator)
        except Exception as e:
            print(f"[error] {src.name}: {e}", file=sys.stderr)
            continue
        out_path.write_text(json.dumps(translated, ensure_ascii=False, indent=2))
        done += 1
        print(f"[done ] {src.name} ({n_chars} chars NL -> {len(translated['body'])} chars EN)")
        time.sleep(args.delay)

    print(f"[translate] translated {done} new articles -> {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
