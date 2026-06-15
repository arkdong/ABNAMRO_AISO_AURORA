"""Intent classification for the standalone tool server."""

from __future__ import annotations

import json
import re
from typing import Any

from .profiles import load_profiles
from .schemas import IntentResult, TaskCode

TMT_SECTOR = "Technologie, Media & Telecom"

_TASK_HINTS: list[tuple[TaskCode, tuple[str, ...]]] = [
    ("T1_TRANSLATE", ("translate", "vertaal", "translation", "naar het nederlands", "naar engels")),
    (
        "T2_COMPLIANCE",
        (
            "check",
            "review",
            "compliance",
            "kpi",
            "checklist",
            "hard rule",
            "quality",
            "controleer",
            "beoordeel",
            "kwaliteit",
            "naleving",
        ),
    ),
    ("T4_RENEWAL", ("old article", "older than", "renew", "refresh", "verouderd", "vernieuw", "actualiseer")),
    ("T1_SEARCH", ("search", "find", "related", "corpus", "exist", "already have", "zoek", "vind", "gerelateerd")),
    ("T1_DRAFT", ("write", "draft", "article", "schrijf", "generate", "concept", "artikel")),
]

_SECTOR_HINTS = (
    "tmt",
    "cyber",
    "agentic",
    "ai",
    "artificial intelligence",
    "kunstmatige intelligentie",
    "telecom",
    "media",
    "advertising",
    "advertentie",
    "buitenreclame",
    "retail media",
    "digital services act",
    "dsa",
    "cloud",
    "saas",
    "software",
    "datacenter",
    "chip",
    "it-leverancier",
    "it supplier",
)

_LANGUAGE_WORDS = {
    "dutch": "nl",
    "nederlands": "nl",
    "nederlandstalig": "nl",
    "english": "en",
    "engels": "en",
}

_BOTH_LANGUAGE_RE = re.compile(
    r"\b(?:both\s+english\s+and\s+dutch|both\s+dutch\s+and\s+english|"
    r"english\s+and\s+dutch|dutch\s+and\s+english|"
    r"engels\s+en\s+nederlands|nederlands\s+en\s+engels|en\s*\+\s*nl)\b"
)

_TARGET_LANGUAGE_PATTERNS = (
    re.compile(
        r"\b(?:final\s+)?(?:draft|output|answer|response|text|article|content)"
        r"\s+(?:must\s+be|should\s+be|needs\s+to\s+be|has\s+to\s+be|is|in)"
        r"\s+(?:written\s+)?(?:in\s+)?"
        r"(?P<lang>dutch|nederlands|nederlandstalig|english|engels)\b"
    ),
    re.compile(
        r"\b(?:write|draft|generate|respond|answer|translate|vertaal|schrijf)"
        r"\b.{0,80}?\b(?:in|to|naar|naar\s+het|in\s+het)\s+"
        r"(?P<lang>dutch|nederlands|nederlandstalig|english|engels)\b"
    ),
)


def _contains(text: str, needle: str) -> bool:
    return needle in text


def _detect_task_codes(prompt_lower: str) -> list[TaskCode]:
    matches: list[TaskCode] = []
    for code, hints in _TASK_HINTS:
        if any(_contains(prompt_lower, hint) for hint in hints):
            matches.append(code)
    return matches or ["T1_DRAFT"]


def _detect_sector(prompt_lower: str) -> str | None:
    if any(hint in prompt_lower for hint in _SECTOR_HINTS):
        return TMT_SECTOR
    return None


def _detect_target_language(prompt_lower: str) -> str | None:
    padded = f" {prompt_lower} "
    if _BOTH_LANGUAGE_RE.search(prompt_lower) or " both " in padded:
        return "both"

    target_language_matches: list[tuple[int, str]] = []
    for pattern in _TARGET_LANGUAGE_PATTERNS:
        for match in pattern.finditer(prompt_lower):
            language = _LANGUAGE_WORDS.get(match.group("lang"))
            if language:
                target_language_matches.append((match.start(), language))
    if target_language_matches:
        return sorted(target_language_matches, key=lambda item: item[0])[-1][1]
    return None


def _detect_language(prompt_lower: str) -> str | None:
    target_language = _detect_target_language(prompt_lower)
    if target_language:
        return target_language

    padded = f" {prompt_lower} "
    nl_hits = sum(
        1
        for signal in (
            "vertaal",
            "schrijf",
            "artikel",
            "artikelen",
            "ouder dan",
            "controleer",
            "zoek",
            "vind",
            " het ",
            " de ",
            " een ",
        )
        if signal in padded
    )
    en_hits = sum(
        1
        for signal in ("write", "translate", "related", "draft", " the ", " an ")
        if signal in padded
    )
    if nl_hits >= 2 and nl_hits > en_hits:
        return "nl"
    if en_hits >= 2 and en_hits > nl_hits:
        return "en"
    return None


def _detect_keywords(prompt_lower: str) -> list[str]:
    bundle = load_profiles()
    seen: set[str] = set()
    out: list[str] = []
    for profile in bundle.domain_expert:
        for keyword in profile.topic_keywords:
            kw_lower = keyword.lower()
            if kw_lower in seen:
                continue
            if kw_lower in prompt_lower:
                seen.add(kw_lower)
                out.append(keyword)
                continue
            parts = [p for p in re.split(r"[^a-z0-9]+", kw_lower) if len(p) >= 3]
            if parts and all(part in prompt_lower for part in parts[:2]):
                seen.add(kw_lower)
                out.append(keyword)
    return out


def _deterministic(prompt: str) -> IntentResult:
    prompt_lower = prompt.lower()
    task_codes = _detect_task_codes(prompt_lower)
    sector = _detect_sector(prompt_lower)
    keywords = _detect_keywords(prompt_lower)
    language = _detect_language(prompt_lower)
    confidence = 0.86 if task_codes[0] != "T1_DRAFT" or "write" in prompt_lower else 0.72
    reason = f"Detected {task_codes[0]} from keyword and task-shape heuristics."
    return IntentResult(
        role="Insights Editorial",
        task_codes=task_codes,
        confidence=confidence,
        task_reason=reason,
        sector=sector,
        topic_keywords=keywords,
        language=language,  # type: ignore[arg-type]
        source="deterministic",
    )


_SYSTEM_PROMPT = """Classify this ABN AMRO editorial request.

Return JSON with: role, task_codes, confidence, task_reason, sector,
topic_keywords, language. Supported task_codes are T1_DRAFT, T1_TRANSLATE,
T1_SEARCH, T2_COMPLIANCE, T4_RENEWAL. Supported sector is
"Technologie, Media & Telecom" or null. language is "en", "nl", "both", or null.
The language field is the requested final output language, not merely the
language of the user's prompt or topic. If the prompt mentions one language
early but later says the final draft/output/answer should be in another
language, use the later final-output instruction.
"""


def _parse_llm_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    return json.loads(raw)


def classify_intent(
    prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> IntentResult:
    if api_key and model:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            parsed = _parse_llm_json(content)
            target_language = _detect_target_language(prompt.lower())
            if target_language:
                parsed["language"] = target_language
            return IntentResult(**parsed, source="llm")
        except Exception:
            pass
    return _deterministic(prompt)
