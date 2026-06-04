"""OpenAI-backed Dutch→English translator for ABN AMRO Insights articles.

Reads `data/writing_guide_rules.md` once and embeds it in a stable system
prompt so OpenAI's automatic prompt caching kicks in across calls.

A single chat call returns structured JSON (`title_en`, `description_en`,
`body_en`) so we can rebuild the EN markdown file in one round-trip.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from pydantic import BaseModel, Field

# Default model. Override via OpenAITranslator(model=…) or CLI --model.
DEFAULT_MODEL = "gpt-5"

# Translation uses a dedicated key so its spend is billed/tracked separately
# from other AURORA components that may also call OpenAI. Falls back to the
# generic OPENAI_API_KEY if the dedicated one isn't set.
TRANSLATION_KEY_ENV = "OPENAI_API_KEY_TRANSLATION"
FALLBACK_KEY_ENV = "OPENAI_API_KEY"

_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "writing_guide_rules.md"


def resolve_api_key() -> str | None:
    """Return the API key for translation, preferring the dedicated env var."""
    return os.environ.get(TRANSLATION_KEY_ENV) or os.environ.get(FALLBACK_KEY_ENV)


# ---- structured-output schema ----------------------------------------------


class TranslatedArticle(BaseModel):
    """The model's response shape — kept narrow on purpose."""

    title_en: str = Field(description="English translation of the article title.")
    description_en: str = Field(
        description="English translation of the OpenGraph description (frontmatter)."
    )
    body_en: str = Field(
        description=(
            "English translation of the Markdown body. Preserve every "
            "Markdown construct (images, links, headings, lists, iframes) "
            "exactly. Translate anchor text but keep URLs verbatim."
        )
    )


# ---- output validation -----------------------------------------------------


_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\([^)]+\)")
_IFRAME_RE = re.compile(r"<iframe\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


@dataclass
class StructureCheck:
    ok: bool
    issues: list[str]


def check_structure(source_md: str, translated_md: str) -> StructureCheck:
    """Cheap sanity check: counts of structural Markdown constructs should match."""
    issues: list[str] = []

    src_imgs = len(_IMG_RE.findall(source_md))
    tgt_imgs = len(_IMG_RE.findall(translated_md))
    if src_imgs != tgt_imgs:
        issues.append(f"image count {src_imgs}→{tgt_imgs}")

    src_links = len(_LINK_RE.findall(source_md))
    tgt_links = len(_LINK_RE.findall(translated_md))
    # links can drift by ±1 if the model rephrases; flag larger gaps
    if abs(src_links - tgt_links) > 1:
        issues.append(f"link count {src_links}→{tgt_links}")

    src_iframes = len(_IFRAME_RE.findall(source_md))
    tgt_iframes = len(_IFRAME_RE.findall(translated_md))
    if src_iframes != tgt_iframes:
        issues.append(f"iframe count {src_iframes}→{tgt_iframes}")

    src_h = len(_HEADING_RE.findall(source_md))
    tgt_h = len(_HEADING_RE.findall(translated_md))
    if abs(src_h - tgt_h) > 1:
        issues.append(f"heading count {src_h}→{tgt_h}")

    src_len = len(source_md)
    tgt_len = len(translated_md)
    if src_len > 0:
        ratio = tgt_len / src_len
        if ratio < 0.5 or ratio > 1.6:
            issues.append(f"length ratio {ratio:.2f}× (expected 0.7–1.3×)")

    return StructureCheck(ok=not issues, issues=issues)


# ---- translator ------------------------------------------------------------


class OpenAITranslator:
    """Translate one NL article to EN with retries and structure validation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        rules_path: Path | None = None,
        reasoning_effort: str = "minimal",
        timeout_seconds: float = 180.0,
    ):
        # Explicit api_key wins; otherwise prefer the dedicated translation key
        # over the generic one so translation spend is tracked separately.
        key = api_key or resolve_api_key()
        if not key:
            raise RuntimeError(
                f"No API key found. Set {TRANSLATION_KEY_ENV} (preferred) "
                f"or {FALLBACK_KEY_ENV} in your environment or .env file."
            )
        # Per-call timeout: GPT-5 with default reasoning can otherwise hang for
        # several minutes on long inputs with no visible progress.
        self.client = openai.OpenAI(api_key=key, timeout=timeout_seconds)
        self.model = model
        # Translation does not need reasoning. "minimal" keeps thinking tokens
        # near zero so the response starts streaming back almost immediately.
        self.reasoning_effort = reasoning_effort
        self.rules = (rules_path or _RULES_PATH).read_text(encoding="utf-8")
        self.last_usage: dict[str, int] | None = None

    def _supports_reasoning(self) -> bool:
        m = self.model.lower()
        return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")

    def _system_prompt(self) -> str:
        return (
            "You are a professional Dutch-to-English translator for ABN AMRO "
            "Insights articles. Translate into British English following the "
            "rules below precisely. Output strictly the structured fields the "
            "user asks for — no commentary, no apologies, no extra prose.\n\n"
            "=== ABN AMRO Writing Guide 2026 V1.1 — Translator Checklist ===\n\n"
            f"{self.rules}\n\n"
            "=== End of checklist ===\n"
        )

    def _user_prompt(self, title_nl: str, description_nl: str, body_nl: str) -> str:
        return (
            "Translate the following Dutch article into British English. "
            "Return three fields: `title_en`, `description_en`, `body_en`. "
            "Preserve every Markdown construct in the body exactly.\n\n"
            f"## title_nl\n{title_nl}\n\n"
            f"## description_nl\n{description_nl}\n\n"
            f"## body_nl\n{body_nl}\n"
        )

    def translate(
        self,
        title_nl: str,
        description_nl: str,
        body_nl: str,
        max_retries: int = 3,
    ) -> TranslatedArticle:
        """Run one translation call with exponential backoff on transient errors."""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {
                            "role": "user",
                            "content": self._user_prompt(title_nl, description_nl, body_nl),
                        },
                    ],
                    response_format=TranslatedArticle,
                )
                # `reasoning_effort` is only valid on reasoning-capable models
                # (gpt-5 family, o-series). Don't pass it to others.
                if self._supports_reasoning():
                    kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self.client.beta.chat.completions.parse(**kwargs)
                if resp.usage:
                    self.last_usage = {
                        "input": resp.usage.prompt_tokens,
                        "output": resp.usage.completion_tokens,
                        "cached": getattr(
                            getattr(resp.usage, "prompt_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0,
                    }
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise RuntimeError("Model returned no parsed structured output")
                return parsed
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                last_err = e
                wait = 2**attempt
                time.sleep(wait)
            except openai.APIStatusError as e:
                # 5xx: retry; 4xx: surface immediately
                if 500 <= e.status_code < 600:
                    last_err = e
                    time.sleep(2**attempt)
                else:
                    raise
        raise RuntimeError(f"Translation failed after {max_retries} retries: {last_err}")
