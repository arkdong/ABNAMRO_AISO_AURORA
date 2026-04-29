"""Translate Dutch article bodies to English with deep-translator.

Reads data/raw/*.json, writes data/translated/*.json, preserving all metadata
and adding language_original="nl", language_embedded="en".
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "translated"


def translate_text(text: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="nl", target="en").translate(text)


def translate_article(article: dict) -> dict:
    return {
        **article,
        "body": translate_text(article["body"]),
        "language_original": "nl",
        "language_embedded": "en",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in RAW_DIR.glob("*.json"):
        article = json.loads(src.read_text())
        translated = translate_article(article)
        (OUT_DIR / src.name).write_text(
            json.dumps(translated, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
