"""Intent classification — rich result for downstream profile selection.

Two entry points:

- :func:`classify_full` — returns ``(IntentResult, source)``. ``IntentResult``
  carries everything ``profiles.match()`` needs: ``task_codes`` (multi-intent),
  ``sector``, ``topic_keywords``, plus ``language`` for prompt assembly and
  ``role`` / ``confidence`` / ``task_reason`` for the UI.
- :func:`classify` — legacy 6-tuple shim; preserved so existing callers don't
  break. Internally projects from :func:`classify_full`.

If both an API key and a model are configured, the LLM path is used with the
configured model. On any failure — or if either field is missing — it
delegates to ``task_definition.IntentClassifier``'s deterministic fallback,
then enriches the result with sector / topic-keyword / language heuristics so
profile matching still has something to work with offline.
"""

from __future__ import annotations

from typing import Literal, Optional

import openai
from loguru import logger
from pydantic import BaseModel, Field

from task_definition.intent_classifier import IntentClassifier

# Constrained sector vocabulary. The only sector with profiles in the repo
# today is TMT — extend this Literal as more sector profiles land.
Sector = Literal["Technologie, Media & Telecom"]
Language = Literal["en", "nl", "both"]


class IntentResult(BaseModel):
    """Everything the next pipeline stage (profile selection) needs."""

    role: str
    task_codes: list[str] = Field(min_length=1)
    confidence: float
    task_reason: str
    sector: Optional[Sector] = None
    topic_keywords: list[str] = Field(default_factory=list)
    language: Optional[Language] = None

    @property
    def task_code(self) -> str:
        """Primary intent — back-compat shim for callers that expect a single code."""
        return self.task_codes[0]


_SYSTEM_PROMPT = """You are an intent classifier for ABN AMRO's editorial co-pilot.

Return a structured result with these fields:

# task_codes — list, primary first
T1_DRAFT: Draft new content
T1_TRANSLATE: Translate existing content
T1_SEARCH: Search corpus for related articles
T2_COMPLIANCE: Quality & compliance check
T4_RENEWAL: Detect & renew aging articles

If the request asks for multiple things (e.g. translate AND show related), include all relevant codes ordered by primary first.

# role
Choose one: Insights Editorial, Chatbot (Anna), Mobile App (UX), Web / IB.

# sector
If the request is clearly about a specific business sector, set `sector`. The only currently supported value is "Technologie, Media & Telecom" (TMT — software, AI, cybersecurity, telecom, media, advertising). For anything else, set sector to null.

# topic_keywords
Emit a small list of lowercase keywords capturing the request's domain (e.g. "agentic ai", "cybersecurity", "retail media", "digital services act"). These are matched case-insensitively against profile topic vocabularies, so prefer concise canonical forms.

# language
The OUTPUT language requested: "en" (English), "nl" (Dutch), "both" if explicitly requested, or null if unclear.

# confidence
Float between 0.0 and 1.0 reflecting certainty about the primary task code.

# task_reason
One short sentence explaining why this classification fits.
"""


# ── Deterministic enrichment helpers ────────────────────────────────────────

# Sector -> substrings that, if present in the prompt, route to that sector.
_SECTOR_HINTS: dict[Sector, tuple[str, ...]] = {
    "Technologie, Media & Telecom": (
        "tmt", "cyber", "agentic", "ai", "kunstmatige intelligentie",
        "telecom", "media", "advertising", "advertentie", "retail media",
        "dsa", "digital services act", "cloud", "saas", "software",
        "datacenter", "chip", "it-leverancier", "it leverancier",
    ),
}


def _detect_sector(prompt_lower: str) -> Optional[Sector]:
    for sector, hints in _SECTOR_HINTS.items():
        if any(h in prompt_lower for h in hints):
            return sector
    return None


def _detect_topic_keywords(prompt_lower: str) -> list[str]:
    """Return profile-vocabulary keywords whose lowercase form appears in the prompt.

    Loaded from the live profile bundle so this stays in sync with whatever
    domain experts are on disk — no hardcoded list to drift.
    """
    from profiles import load_all  # imported lazily to avoid circulars

    bundle = load_all()
    seen: set[str] = set()
    matches: list[str] = []
    for expert in bundle.domain_expert:
        for kw in expert.topic_keywords:
            kw_lower = kw.lower()
            if kw_lower in seen:
                continue
            if kw_lower in prompt_lower:
                matches.append(kw)
                seen.add(kw_lower)
    return matches


def _detect_language(prompt_lower: str) -> Optional[Language]:
    """Lightweight NL/EN detection. Padded substrings to avoid false positives."""
    nl_signals = (
        "vertaal", "schrijf", "gerelateerd", "artikelen", "ouder dan",
        " het ", " de ", " een ", "voor de", "naar het",
    )
    en_signals = (
        "write", "translate", "older than", "related", "draft",
        " the ", " a ", " an ", "in english",
    )
    padded = f" {prompt_lower} "
    nl_hits = sum(1 for s in nl_signals if s in padded)
    en_hits = sum(1 for s in en_signals if s in padded)
    if nl_hits >= 2 and nl_hits > en_hits:
        return "nl"
    if en_hits >= 2 and en_hits > nl_hits:
        return "en"
    if nl_hits >= 2 and en_hits >= 2:
        return "both"
    return None


def _deterministic(prompt: str) -> IntentResult:
    """Run the teammate's deterministic classifier and enrich with heuristics."""
    role, task_code, confidence, reason, _ = IntentClassifier().classify(prompt, api_key=None)
    p = prompt.lower()
    return IntentResult(
        role=role,
        task_codes=[task_code],
        confidence=confidence,
        task_reason=reason,
        sector=_detect_sector(p),
        topic_keywords=_detect_topic_keywords(p),
        language=_detect_language(p),
    )


# ── Public API ──────────────────────────────────────────────────────────────


def classify_full(
    prompt: str,
    api_key: str | None,
    model: str | None,
) -> tuple[IntentResult, Literal["llm", "deterministic"]]:
    """Classify ``prompt`` and return the rich result plus its source."""
    if api_key and model:
        try:
            logger.info(f"Backend intent: classifying via LLM (model={model})...")
            client = openai.OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=IntentResult,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("LLM returned no parsed result")
            return parsed, "llm"
        except Exception as e:
            logger.warning(f"Backend LLM classify failed ({e}); falling back to deterministic")

    return _deterministic(prompt), "deterministic"


def classify(
    prompt: str,
    api_key: str | None,
    model: str | None,
) -> tuple[str, str, float, str, str | None, str]:
    """Legacy 6-tuple ``(role, task_code, confidence, reason, raw_json, source)``.

    Kept so any caller still on the old shape doesn't break. New code should
    call :func:`classify_full` and read fields off :class:`IntentResult`.
    """
    result, source = classify_full(prompt, api_key, model)
    raw_json = result.model_dump_json(indent=2) if source == "llm" else None
    return (
        result.role,
        result.task_codes[0],
        result.confidence,
        result.task_reason,
        raw_json,
        source,
    )
