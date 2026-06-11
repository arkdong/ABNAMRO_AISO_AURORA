"""Tier 1 — deterministic content checks.

These run on every generated piece of content before any LLM judge fires.
They are cheap, repeatable and the failures of any Mandatory-Blocking
Tier 1 KPI short-circuits the rest of the pipeline.

Each check is a pure function returning a :class:`KPIResult`. The dispatcher
``run_tier1`` walks the catalogue, picks the checker registered for each
``kpi_id``, and ignores KPIs without a registered check (those fall through
to Tier 2 or stay un-evaluated).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from backend.content_generation.types import ContentRequest, ContentResult
from backend.evaluation.catalogue import KPI, Catalogue
from backend.evaluation.indicators import (
    ErrorScale,
    LanguageLevelScale,
    LengthScale,
    PresenceScale,
    UsedScale,
    YesNoScale,
)
from backend.evaluation.types import Channel, KPIResult, Origin


# ---------- markdown parsing helpers --------------------------------------

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_SENTENCE_END = re.compile(r"[\.\!\?](?:\s|$)")


def _strip_md(text: str) -> str:
    """Cheaply strip markdown to get word-like body text."""
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"[*_>#]+", "", text)
    return text


def _split_sentences(text: str) -> list[str]:
    bare = _strip_md(text)
    parts = [p.strip() for p in _SENTENCE_END.split(bare) if p.strip()]
    return parts


def _split_paragraphs(text: str) -> list[str]:
    bare = _strip_md(text)
    return [p.strip() for p in re.split(r"\n\s*\n", bare) if p.strip()]


def _word_count(text: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", _strip_md(text))])


# ---------- shared result helper ------------------------------------------


def _result(
    kpi: KPI,
    *,
    value: str,
    passed: bool,
    reason: str,
    indicator: Optional[str] = None,
    raw_metric: Optional[dict] = None,
) -> KPIResult:
    return KPIResult(
        kpi_id=kpi.id,
        name=kpi.name,
        cluster=kpi.cluster_short,
        category=kpi.category,
        weight=kpi.weight,  # type: ignore[arg-type]
        monitoring=kpi.monitoring,  # type: ignore[arg-type]
        indicator=indicator or kpi.indicator,
        value=value,
        raw_metric=raw_metric,
        reason=reason,
        tier=1,
        passed=passed,
        source="deterministic",
    )


# ---------- individual checks ---------------------------------------------


def check_sentence_length(
    kpi: KPI, req: ContentRequest, gen: ContentResult, *, max_words: int = 15
) -> KPIResult:
    """B1: max 15 words per sentence. (Readability cluster.)"""
    sentences = _split_sentences(gen.body)
    if not sentences:
        return _result(
            kpi, value=LengthScale.unknown.value, passed=False,
            reason="no sentences found in body",
        )
    too_long = [s for s in sentences if len(re.findall(r"\b\w+\b", s)) > max_words]
    ratio = len(too_long) / len(sentences)
    passed = ratio <= 0.10  # tolerate up to 10% long sentences
    return _result(
        kpi,
        value=(LengthScale.right if passed else LengthScale.too_long).value,
        passed=passed,
        reason=f"{len(too_long)}/{len(sentences)} sentences exceed {max_words} words",
        raw_metric={"long": len(too_long), "total": len(sentences), "ratio": ratio},
    )


def check_paragraph_length(
    kpi: KPI, req: ContentRequest, gen: ContentResult, *, max_words: int = 100
) -> KPIResult:
    """Max 100 words per paragraph (B1)."""
    paragraphs = _split_paragraphs(gen.body)
    if not paragraphs:
        return _result(
            kpi, value=LengthScale.unknown.value, passed=False,
            reason="no paragraphs found",
        )
    too_long = [p for p in paragraphs if _word_count(p) > max_words]
    passed = not too_long
    return _result(
        kpi,
        value=(LengthScale.right if passed else LengthScale.too_long).value,
        passed=passed,
        reason=f"{len(too_long)}/{len(paragraphs)} paragraphs exceed {max_words} words",
        raw_metric={"long": len(too_long), "total": len(paragraphs)},
    )


_EN_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", re.IGNORECASE
)
_NL_PASSIVE_RE = re.compile(r"\b(?:worden|wordt|werd|werden|geworden)\b", re.IGNORECASE)


def check_passive_voice(kpi: KPI, req: ContentRequest, gen: ContentResult) -> KPIResult:
    """Best-effort passive-voice detection; B1 target is near-zero."""
    lang = (getattr(req.intent, "language", None) or "en").lower()
    pattern = _NL_PASSIVE_RE if lang.startswith("nl") else _EN_PASSIVE_RE
    hits = pattern.findall(_strip_md(gen.body))
    sentences = max(1, len(_split_sentences(gen.body)))
    ratio = len(hits) / sentences
    passed = ratio <= 0.10
    return _result(
        kpi,
        value=(PresenceScale.not_present if passed else PresenceScale.present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"{len(hits)} passive markers across {sentences} sentences ({ratio:.0%})",
        raw_metric={"hits": len(hits), "sentences": sentences},
    )


def check_bullet_list_presence(
    kpi: KPI, req: ContentRequest, gen: ContentResult
) -> KPIResult:
    """Norm: at least one bullet list in body for scannability."""
    hits = len(_BULLET_RE.findall(gen.body))
    passed = hits >= 1
    return _result(
        kpi,
        value=(PresenceScale.present if passed else PresenceScale.not_present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"{hits} bullet list lines detected",
        raw_metric={"count": hits},
    )


def check_h1_count(kpi: KPI, req: ContentRequest, gen: ContentResult) -> KPIResult:
    """Exactly one H1 (the page title)."""
    h1s = _H1_RE.findall(gen.body)
    passed = len(h1s) == 1
    return _result(
        kpi,
        value=(PresenceScale.present if passed else PresenceScale.not_present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"{len(h1s)} H1 headings (expected 1)",
        raw_metric={"count": len(h1s)},
    )


def check_keyword_in_h1(kpi: KPI, req: ContentRequest, gen: ContentResult) -> KPIResult:
    """Top topic keyword must appear in the H1."""
    keywords = [k.lower() for k in (req.intent.topic_keywords or []) if k]
    if not keywords:
        return _result(
            kpi,
            value=PresenceScale.unknown.value,
            indicator="PresenceScale",
            passed=False,
            reason="no topic keywords on intent",
        )
    h1s = _H1_RE.findall(gen.body)
    text = " ".join(h1s).lower()
    hit = next((k for k in keywords if k in text), None)
    passed = bool(hit)
    return _result(
        kpi,
        value=(PresenceScale.present if passed else PresenceScale.not_present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"keyword in H1: {hit!r}" if hit else "no topic keyword found in H1",
        raw_metric={"h1s": h1s, "keywords": keywords},
    )


def check_images_alt_present(
    kpi: KPI, req: ContentRequest, gen: ContentResult
) -> KPIResult:
    """Every image must carry non-empty alt text. WCAG basic requirement."""
    images = _IMAGE_RE.findall(gen.body)
    if not images:
        return _result(
            kpi,
            value=PresenceScale.present.value,
            indicator="PresenceScale",
            passed=True,
            reason="no images in body (vacuous pass)",
            raw_metric={"images": 0},
        )
    missing = [url for alt, url in images if not alt.strip()]
    passed = not missing
    return _result(
        kpi,
        value=(PresenceScale.present if passed else PresenceScale.not_present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"{len(missing)}/{len(images)} images missing alt text",
        raw_metric={"missing": missing, "total": len(images)},
    )


def check_tracability(kpi: KPI, req: ContentRequest, gen: ContentResult) -> KPIResult:
    """Generated content must echo source IDs + version tags (Blocking).

    Today the ``ContentResult.citations`` list carries ``source_doc``/``node_id``
    pairs back to the corpus snippets, which is the operational stand-in for
    "source ID + version tag" until a versioned source manifest lands.
    """
    if not gen.citations:
        return _result(
            kpi,
            value=UsedScale.not_used.value,
            passed=False,
            reason="generated content carries no source citations",
        )
    seen = {(c.source_doc, c.node_id) for c in gen.citations}
    return _result(
        kpi,
        value=UsedScale.used.value,
        passed=True,
        reason=f"{len(seen)} unique source ID(s) cited",
        raw_metric={"unique_sources": len(seen), "citations": len(gen.citations)},
    )


def check_approved_source_for_genai(
    kpi: KPI, req: ContentRequest, gen: ContentResult
) -> KPIResult:
    """No cited snippet is marked as excluded for GenAI use (Blocking).

    Until snippet provenance carries an explicit ``exclude_for_genai`` flag we
    pass vacuously: there's nothing in the current corpus that flags itself
    as excluded. The check is wired up so the data model can be added without
    revisiting this module.
    """
    excluded = []
    for c in gen.citations:
        # Best-effort lookup against ``req.snippets`` since ``Citation`` only
        # carries the identifying fields.
        snip = next(
            (
                s
                for s in req.snippets
                if s.source_doc == c.source_doc and s.node_id == c.node_id
            ),
            None,
        )
        if snip is None:
            continue
        if getattr(snip, "exclude_for_genai", False):
            excluded.append(f"{c.source_doc}::{c.node_id}")
    passed = not excluded
    from backend.evaluation.indicators import ExclusionScale

    return _result(
        kpi,
        value=(ExclusionScale.no_exclusion if passed else ExclusionScale.exclusion).value,
        indicator="ExclusionScale",
        passed=passed,
        reason="all cited sources approved" if passed else f"{len(excluded)} excluded source(s) cited: {excluded}",
        raw_metric={"excluded": excluded},
    )


def check_factuality_no_hallucinated_citations(
    kpi: KPI, req: ContentRequest, gen: ContentResult
) -> KPIResult:
    """Deterministic floor for the Factuality KPI: every ``[n]`` marker in the
    body must correspond to a real citation in the snippet list.

    The full factuality judgement happens in Tier 2. This check exists so the
    short-circuit gate catches obvious citation-fabrication without an LLM
    call. If it passes the Tier 2 judge then refines the verdict; if it
    fails, generation is short-circuited as a Blocking failure.
    """
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", gen.body)}
    n = len(req.snippets)
    invalid = {m for m in markers if m < 1 or m > n}
    passed = not invalid
    if passed:
        # Defer the hard Factuality judgement to Tier 2 — emit a clean
        # "no obvious hallucinations" signal but keep the result tier-1.
        return _result(
            kpi,
            value=ErrorScale.none_.value,
            indicator="ErrorScale",
            passed=True,
            reason="no out-of-range citation markers (deterministic floor only)",
            raw_metric={"markers": sorted(markers), "snippet_count": n},
        )
    return _result(
        kpi,
        value=ErrorScale.several.value,
        indicator="ErrorScale",
        passed=False,
        reason=f"hallucinated citation indices: {sorted(invalid)}",
        raw_metric={"invalid": sorted(invalid), "snippet_count": n},
    )


def check_reading_level_b1(
    kpi: KPI, req: ContentRequest, gen: ContentResult
) -> KPIResult:
    """Rough Flesch reading-ease → CEFR band mapping; B1 target."""
    sentences = _split_sentences(gen.body)
    words = re.findall(r"\b[\w']+\b", _strip_md(gen.body))
    if not sentences or not words:
        return _result(
            kpi,
            value=LanguageLevelScale.unknown.value,
            passed=False,
            reason="body too short to measure reading level",
        )
    avg_sentence_len = len(words) / len(sentences)
    syllables = sum(_count_syllables(w) for w in words)
    avg_syllables_per_word = syllables / max(1, len(words))
    # Flesch reading ease (English-tuned, used as a coarse proxy for NL too).
    score = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word
    band = (
        LanguageLevelScale.A1 if score >= 90 else
        LanguageLevelScale.A2 if score >= 80 else
        LanguageLevelScale.B1 if score >= 60 else
        LanguageLevelScale.B2 if score >= 50 else
        LanguageLevelScale.C1 if score >= 30 else
        LanguageLevelScale.C2
    )
    passed = band in (LanguageLevelScale.A1, LanguageLevelScale.A2, LanguageLevelScale.B1)
    return _result(
        kpi,
        value=band.value,
        indicator="LanguageLevelScale",
        passed=passed,
        reason=f"Flesch score {score:.1f} → {band.value}",
        raw_metric={
            "flesch": round(score, 1),
            "avg_sentence_len": round(avg_sentence_len, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
        },
    )


_VOWEL_GROUP = re.compile(r"[aeiouyAEIOUY]+")


def _count_syllables(word: str) -> int:
    word = word.lower()
    if not word:
        return 0
    groups = _VOWEL_GROUP.findall(word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


# ---------- dispatcher ----------------------------------------------------

CheckFn = Callable[[KPI, ContentRequest, ContentResult], KPIResult]

# Map ``kpi.id`` (the slug from the catalogue) → checker. KPIs missing from
# this registry get no Tier 1 result; they're picked up by Tier 2 or stay
# un-evaluated (the service marks those as ``skipped``).
CHECK_REGISTRY: dict[str, CheckFn] = {
    # ── Readability cluster (Accessibility & inclusion / readability) ──
    "sentence_number_of_words": check_sentence_length,
    "paragraph_bubble_number_of_words_sentences": check_paragraph_length,
    "sentence_structure": check_passive_voice,
    "reading_level": check_reading_level_b1,
    # ── Structuring & design cluster ───────────────────────────────────
    "bullet_list_points": check_bullet_list_presence,
    "images_with_missing_alt_text": check_images_alt_present,
    # ── Content structure / SEO floors ─────────────────────────────────
    "h1_header_presence": check_h1_count,
    "h1_header_keywords": check_keyword_in_h1,
    # ── Compliancy & substantive quality (Blocking floors) ─────────────
    "tracability": check_tracability,
    "approved_source_content_for_genai": check_approved_source_for_genai,
    "factuality_truthfullness": check_factuality_no_hallucinated_citations,
}


def run_tier1(
    *,
    catalogue: Catalogue,
    req: ContentRequest,
    gen: ContentResult,
    origin: Origin,
    channel: Channel,
) -> list[KPIResult]:
    """Run every registered Tier 1 check that applies to this content."""
    applicable = catalogue.applicable(origin=origin, channel=channel)
    out: list[KPIResult] = []
    for kpi in applicable:
        fn = CHECK_REGISTRY.get(kpi.id)
        if fn is None:
            continue
        try:
            out.append(fn(kpi, req, gen))
        except Exception as e:  # noqa: BLE001 — checks must never break the pipeline
            out.append(
                _result(
                    kpi,
                    value="unknown",
                    passed=False,
                    reason=f"check error: {e}",
                )
            )
    return out
