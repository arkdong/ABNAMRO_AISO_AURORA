"""Multi-method Dutch->English translator for AURORA articles.

Reads `data/raw/*.md` (YAML frontmatter + Markdown body) and writes translated
copies to `data/translated/<variant>/`, preserving Markdown structure.
The variant subfolder is `<method>` for Google or `<method>-<model>` for LLMs,
so multiple methods can sit side-by-side and be compared.

Supported backends:
  - google     deep-translator + Google Translate (free, no API key)
  - anthropic  Claude API (requires ANTHROPIC_API_KEY)
                 default: claude-sonnet-4-5
                 also:    claude-haiku-4-5
  - openai     OpenAI API (requires OPENAI_API_KEY)
                 default: gpt-4o
                 also:    gpt-4o-mini

Each translated file gets these frontmatter fields added/updated:
  language:           "en"
  language_original:  "nl"
  translated_with:    "<method>" or "<method>:<model>"
  source_url:         (the original article URL — preserved)
  title_original:     original Dutch title

The original Dutch body is NOT embedded in the output (that's data/raw/<slug>.md).
This keeps each file focused and lets us swap the body for retrieval cleanly.

Usage:
    python -m scripts.translate --method google
    python -m scripts.translate --method anthropic --model claude-sonnet-4-5
    python -m scripts.translate --method anthropic --model claude-haiku-4-5
    python -m scripts.translate --method openai --model gpt-4o
    python -m scripts.translate --method openai --model gpt-4o-mini

    # Smoke-test on 3 articles before paying for full corpus
    python -m scripts.translate --method anthropic --limit 3

    # Re-translate everything (ignore skip-if-exists)
    python -m scripts.translate --method google --force
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Protocol

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
OUT_BASE = ROOT_DIR / "data" / "translated"

# Google Translate hard limit is 5000 chars; pad below for safety.
GOOGLE_MAX_CHARS = 4500


# ---------------------------------------------------------------------------
# System prompt for LLM translators
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """You are a professional Dutch-to-English translator specializing in financial, banking, and business journalism for the Dutch market (ABN AMRO insight articles).

TRANSLATION RULES — follow exactly:

1. Preserve ALL Markdown formatting verbatim:
   - Headings (`#`, `##`, `###`)
   - Bullet lists (`-`)
   - Bold (`**...**`) and italic (`*...*`)
   - Block quotes (`> ...`)
   - Inline links — translate the link text but keep the URL unchanged: `[translated text](https://original.url)`
2. Do NOT translate:
   - URLs
   - Code blocks or inline code
   - Proper nouns (company names, product names, person names — e.g. "Wageningen University & Research", "ABN AMRO", "HEMA")
   - Currency symbols, numbers, percentages, dates — keep them exact
3. Translate idiomatically and fluently. Write natural English a Dutch business reader would expect to see in an English-language version of the same publication. Avoid literal word-for-word translation.
4. Preserve paragraph breaks (blank lines between paragraphs).
5. OUTPUT ONLY the translated Markdown. Do NOT add a preamble, explanation, quotation marks around the result, "Here is the translation:", or any other commentary. Your entire output is the translation, nothing else."""


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Translator backend. translate() takes Dutch text and returns English."""

    name: str  # used for output subdir naming

    def translate(self, text: str) -> str: ...


# ---------------------------------------------------------------------------
# Google (deep-translator)
# ---------------------------------------------------------------------------


def _batch_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split text into batches of paragraphs each <= max_chars."""
    paragraphs = text.split("\n\n")
    batches: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if len(p) > max_chars:
            # paragraph itself too long — hard-split on newlines
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


class GoogleBackend:
    """deep-translator + Google Translate. Free, no API key, but lossy on
    idiom, occasionally mangles markdown links, and rate-limits at scale."""

    name = "deep-translator"

    def __init__(self) -> None:
        from deep_translator import GoogleTranslator

        self._translator = GoogleTranslator(source="nl", target="en")

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        batches = _batch_paragraphs(text, GOOGLE_MAX_CHARS)
        out: list[str] = []
        for b in batches:
            if not b.strip():
                continue
            out.append(self._translator.translate(b))
        return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicBackend:
    """Claude API translator. Uses prompt caching on the system prompt so
    translating many articles in one run amortizes the system-prompt tokens."""

    SUPPORTED_MODELS = ("claude-sonnet-4-5", "claude-haiku-4-5")
    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if model not in self.SUPPORTED_MODELS:
            print(
                f"[anthropic] warning: model {model!r} not in tested list "
                f"{self.SUPPORTED_MODELS}",
                file=sys.stderr,
            )
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise SystemExit("Install anthropic: pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set in environment.")
        self._client = Anthropic()
        self._model = model
        self.name = f"anthropic-{model}"

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        # Cache the system prompt so subsequent articles get a cache hit.
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": LLM_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": text}],
        )
        # Concatenate any text blocks (usually just one)
        return "".join(b.text for b in msg.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIBackend:
    """OpenAI Chat Completions translator. OpenAI auto-caches prompts >1024
    tokens server-side, so the system prompt benefits from caching after the
    first call without any explicit cache_control."""

    SUPPORTED_MODELS = ("gpt-4o", "gpt-4o-mini")
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if model not in self.SUPPORTED_MODELS:
            print(
                f"[openai] warning: model {model!r} not in tested list "
                f"{self.SUPPORTED_MODELS}",
                file=sys.stderr,
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("Install openai: pip install openai") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set in environment.")
        self._client = OpenAI()
        self._model = model
        self.name = f"openai-{model}"

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Frontmatter I/O
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def read_md(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: no YAML frontmatter")
    frontmatter = yaml.safe_load(m.group(1)) or {}
    body = text[m.end() :].lstrip("\n")
    return frontmatter, body


def write_md(path: Path, frontmatter: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_block = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    path.write_text(f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_backend(method: str, model: str | None) -> Backend:
    if method == "google":
        return GoogleBackend()
    if method == "anthropic":
        return AnthropicBackend(model or AnthropicBackend.DEFAULT_MODEL)
    if method == "openai":
        return OpenAIBackend(model or OpenAIBackend.DEFAULT_MODEL)
    raise SystemExit(f"unknown method: {method!r}")


def translate_article(frontmatter: dict, body: str, backend: Backend) -> tuple[dict, str]:
    title_nl = frontmatter.get("title")
    body_en = backend.translate(body)
    title_en = backend.translate(title_nl) if title_nl else None

    new_fm = dict(frontmatter)
    new_fm["title"] = title_en or title_nl
    if title_nl:
        new_fm["title_original"] = title_nl
    new_fm["language"] = "en"
    new_fm["language_original"] = "nl"
    new_fm["translated_with"] = backend.name
    # preserve source url under a stable key
    if "url" in new_fm and "source_url" not in new_fm:
        new_fm["source_url"] = new_fm["url"]
    return new_fm, body_en


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        choices=("google", "anthropic", "openai"),
        required=True,
    )
    ap.add_argument(
        "--model",
        default=None,
        help="model name for anthropic/openai (ignored for google)",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap for testing")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between articles")
    ap.add_argument("--force", action="store_true", help="re-translate existing files")
    args = ap.parse_args()

    backend = make_backend(args.method, args.model)
    out_dir = OUT_BASE / backend.name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.md"))
    print(f"[translate] backend={backend.name} src={RAW_DIR} out={out_dir}", file=sys.stderr)
    print(f"[translate] {len(raw_files)} source articles", file=sys.stderr)

    done = 0
    skipped = 0
    errors = 0
    for src in raw_files:
        if args.limit and done >= args.limit:
            break
        out_path = out_dir / src.name
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            frontmatter, body = read_md(src)
        except Exception as e:
            print(f"[error] {src.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        n_chars = len(body)
        try:
            new_fm, body_en = translate_article(frontmatter, body, backend)
        except Exception as e:
            print(f"[error] {src.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        write_md(out_path, new_fm, body_en)
        done += 1
        print(f"[done {done:>3}] {src.name} ({n_chars} -> {len(body_en)} chars)")
        time.sleep(args.delay)

    print(
        f"[translate] done={done} skipped={skipped} errors={errors} -> {out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
