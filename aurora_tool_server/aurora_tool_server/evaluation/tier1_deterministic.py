"""Tier 1 — deterministic content checks.

These run before any LLM judge fires. The default generation stage keeps the
active registry small and objective; broader readability/SEO/style checkers
remain in this module for audit workflows, but they are not part of the
default pass/fail gate.

Each check is a pure function returning a :class:`KPIResult`. The dispatcher
``run_tier1`` walks the catalogue, picks the checker registered for each
``kpi_id``, and ignores KPIs without a registered check (those fall through
to Tier 2 or stay un-evaluated).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from ..schemas import Channel, ContentResult, KPIResult, Origin
from .catalogue import KPI, Catalogue
from .indicators import (
    DeviationYesNo,
    ErrorScale,
    ExclusionScale,
    LanguageLevelScale,
    LengthScale,
    PresenceScale,
    UsedScale,
)
from .inputs import EvalRequest


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
    source: str = "deterministic",
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
        source=source,  # type: ignore[arg-type]
    )


# ---------- individual checks ---------------------------------------------


def check_sentence_length(
    kpi: KPI, req: EvalRequest, gen: ContentResult, *, max_words: int = 15
) -> KPIResult:
    """Workbook 01.02.05: "min. 5 words, max. 20 words/B1: max. 15 words";
    chat deviant norm "Max 10-12 words per sentence".

    The 10% long-sentence tolerance is engineering calibration, not a
    workbook norm — the workbook states the per-sentence cap only.
    """
    if req.channel == "chat":
        max_words = 12  # chat deviant norm: max 10-12 words per sentence
    min_words = 5
    sentences = _split_sentences(gen.body)
    if not sentences:
        return _result(
            kpi, value=LengthScale.unknown.value, passed=False,
            reason="no sentences found in body",
        )
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    too_long = sum(1 for n in lengths if n > max_words)
    too_short = sum(1 for n in lengths if n < min_words)
    ratio = (too_long + too_short) / len(sentences)
    passed = ratio <= 0.10  # tolerate up to 10% out-of-range sentences
    return _result(
        kpi,
        value=(LengthScale.right if passed else LengthScale.too_long).value,
        passed=passed,
        reason=(
            f"{too_long}/{len(sentences)} sentences exceed {max_words} words, "
            f"{too_short} below {min_words} words"
        ),
        raw_metric={
            "long": too_long,
            "short": too_short,
            "total": len(sentences),
            "ratio": ratio,
            "max_words": max_words,
        },
    )


def check_paragraph_length(
    kpi: KPI, req: EvalRequest, gen: ContentResult, *, max_words: int = 100
) -> KPIResult:
    """Workbook 01.02.02: "max. 100 words per paragraph/3-10 sentences";
    chat deviant norm "max. 160 characters per bubble / max 3 bubbles"."""
    paragraphs = _split_paragraphs(gen.body)
    if not paragraphs:
        return _result(
            kpi, value=LengthScale.unknown.value, passed=False,
            reason="no paragraphs found",
        )
    if req.channel == "chat":
        too_long = [p for p in paragraphs if len(p) > 160]
        passed = not too_long and len(paragraphs) <= 3
        return _result(
            kpi,
            value=(LengthScale.right if passed else LengthScale.too_long).value,
            passed=passed,
            reason=(
                f"{len(too_long)}/{len(paragraphs)} bubbles exceed 160 characters; "
                f"{len(paragraphs)} bubbles (max 3)"
            ),
            raw_metric={"long": len(too_long), "bubbles": len(paragraphs)},
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


def check_passive_voice(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
    """Workbook 06.03.04 "Writing style - active": "B1: max. 0 passive sentences".

    Registered under ``writing_style_active`` (Mandatory/High). The regex is
    best-effort; zero hits is the workbook norm, so any detected passive
    construction fails. Hit count stays in ``raw_metric`` for triage.
    """
    lang = (getattr(req.intent, "language", None) or "en").lower()
    pattern = _NL_PASSIVE_RE if lang.startswith("nl") else _EN_PASSIVE_RE
    hits = pattern.findall(_strip_md(gen.body))
    sentences = max(1, len(_split_sentences(gen.body)))
    passed = len(hits) == 0
    return _result(
        kpi,
        value=(PresenceScale.not_present if passed else PresenceScale.present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"{len(hits)} passive markers across {sentences} sentences (norm: 0)",
        raw_metric={"hits": len(hits), "sentences": sentences},
    )


def _bullet_runs(body: str) -> list[int]:
    """Lengths of contiguous bullet-list runs in the body."""
    runs: list[int] = []
    current = 0
    for line in body.splitlines():
        if re.match(r"^\s*[-*+]\s+\S", line):
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def check_bullet_list_presence(
    kpi: KPI, req: EvalRequest, gen: ContentResult
) -> KPIResult:
    """Workbook 01.03.02: "min. 3 - max. 6 points in bullet list".

    At least one list must be present (Textmetrics floor) and every list
    must carry 3-6 items — the workbook flags the presence-only tool rule
    as "limited match (no norm for number of items)".
    """
    runs = _bullet_runs(gen.body)
    if not runs:
        return _result(
            kpi,
            value=PresenceScale.not_present.value,
            indicator="PresenceScale",
            passed=False,
            reason="no bullet list detected",
            raw_metric={"runs": []},
        )
    out_of_range = [n for n in runs if n < 3 or n > 6]
    passed = not out_of_range
    return _result(
        kpi,
        value=(PresenceScale.present if passed else PresenceScale.not_present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=(
            f"{len(runs)} list(s); all within 3-6 items"
            if passed
            else f"{len(out_of_range)}/{len(runs)} list(s) outside 3-6 items: {out_of_range}"
        ),
        raw_metric={"runs": runs},
    )


def check_h1_count(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
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


def _intent_keywords(req: EvalRequest) -> list[str] | None:
    """Lowercased topic keywords, or ``None`` when unavailable (skip checks)."""
    if req.intent is None:
        return None
    keywords = [k.lower() for k in (req.intent.topic_keywords or []) if k]
    return keywords or None


def _skipped(kpi: KPI, *, indicator: str, reason: str) -> KPIResult:
    return _result(
        kpi,
        value="unknown",
        indicator=indicator,
        passed=True,
        reason=reason,
        source="skipped",
    )


def check_keyword_in_h1(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
    """Workbook 07.05.03: "1-2 keywords" in the H1 header; indicator
    "yes/no deviation from norm".

    Both bounds are enforced — keyword stuffing (3+) deviates just like a
    missing keyword. Without intent keywords the check is skipped rather
    than failed — the draft should not be penalised for missing pipeline
    context.
    """
    keywords = _intent_keywords(req)
    if keywords is None:
        return _skipped(
            kpi, indicator="DeviationYesNo",
            reason="no intent keywords supplied; H1 keyword check skipped",
        )
    h1s = _H1_RE.findall(gen.body)
    text = " ".join(h1s).lower()
    count = sum(1 for k in keywords if k in text)
    passed = 1 <= count <= 2
    return _result(
        kpi,
        value=(DeviationYesNo.no if passed else DeviationYesNo.yes).value,
        indicator="DeviationYesNo",
        passed=passed,
        reason=f"{count} keyword(s) in H1 (norm: 1-2)",
        raw_metric={"h1s": h1s, "keywords": keywords, "count": count},
    )


def check_images_alt_present(
    kpi: KPI, req: EvalRequest, gen: ContentResult
) -> KPIResult:
    """Workbook 01.03.07 "Images with missing alt text": obligatory alt tag.

    The KPI measures the *defect* — workbook ``present`` means missing-alt
    images are present (fail); ``not_present`` means no defect (pass). The
    norm scopes the obligation to informative images; markdown carries no
    informative/decorative distinction, so all images are held to it.
    """
    images = _IMAGE_RE.findall(gen.body)
    if not images:
        return _result(
            kpi,
            value=PresenceScale.not_present.value,
            indicator="PresenceScale",
            passed=True,
            reason="no images in body (vacuous pass)",
            raw_metric={"images": 0},
        )
    missing = [url for alt, url in images if not alt.strip()]
    passed = not missing
    return _result(
        kpi,
        value=(PresenceScale.not_present if passed else PresenceScale.present).value,
        indicator="PresenceScale",
        passed=passed,
        reason=f"{len(missing)}/{len(images)} images missing alt text",
        raw_metric={"missing": missing, "total": len(images)},
    )


def check_tracability(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
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
    kpi: KPI, req: EvalRequest, gen: ContentResult
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
    return _result(
        kpi,
        value=(ExclusionScale.no_exclusion if passed else ExclusionScale.exclusion).value,
        indicator="ExclusionScale",
        passed=passed,
        reason="all cited sources approved" if passed else f"{len(excluded)} excluded source(s) cited: {excluded}",
        raw_metric={"excluded": excluded},
    )


def check_factuality_no_hallucinated_citations(
    kpi: KPI, req: EvalRequest, gen: ContentResult
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
    kpi: KPI, req: EvalRequest, gen: ContentResult
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


def check_text_sentence_count(
    kpi: KPI, req: EvalRequest, gen: ContentResult, *, max_sentences: int = 100
) -> KPIResult:
    """Workbook 01.02.08 "Text - number of sentences": "B1: max. 100 sentences"."""
    n = len(_split_sentences(gen.body))
    passed = n <= max_sentences
    return _result(
        kpi,
        value=(LengthScale.right if passed else LengthScale.too_long).value,
        indicator="LengthScale",
        passed=passed,
        reason=f"{n} sentences (norm: max {max_sentences})",
        raw_metric={"sentences": n},
    )


def check_referrals(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
    """Workbook 07.04.04 "Referrals": "min. 1-3 referrals".

    Counted as markdown links in the body (referrals to related content).
    Citation markers ``[n]`` are evidence references, not referrals, and do
    not count.
    """
    links = _LINK_RE.findall(_IMAGE_RE.sub("", gen.body))
    n = len(links)
    passed = 1 <= n <= 3
    return _result(
        kpi,
        value=(DeviationYesNo.no if passed else DeviationYesNo.yes).value,
        indicator="DeviationYesNo",
        passed=passed,
        reason=f"{n} referral link(s) in body (norm: 1-3)",
        raw_metric={"links": [url for _, url in links]},
    )


def check_h2_h6_count(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
    """Workbook 07.03.02 "H2-6 headers - number": "1-14 headings"."""
    n = len(re.findall(r"^#{2,6}\s+\S", gen.body, re.MULTILINE))
    passed = 1 <= n <= 14
    return _result(
        kpi,
        value=(DeviationYesNo.no if passed else DeviationYesNo.yes).value,
        indicator="DeviationYesNo",
        passed=passed,
        reason=f"{n} H2-H6 heading(s) (norm: 1-14)",
        raw_metric={"count": n},
    )


def check_text_word_count(
    kpi: KPI, req: EvalRequest, gen: ContentResult, *, min_words: int = 300
) -> KPIResult:
    """Workbook 07.02.05 "Text - number of words": "min. 300 words" (web)."""
    n = _word_count(gen.body)
    passed = n >= min_words
    return _result(
        kpi,
        value=(LengthScale.right if passed else LengthScale.too_long).value,
        indicator="LengthScale",
        passed=passed,
        reason=f"{n} words in body (norm: min {min_words})",
        raw_metric={"words": n},
    )


def check_headers_titles_length(
    kpi: KPI, req: EvalRequest, gen: ContentResult
) -> KPIResult:
    """Workbook 01.03.06 "Headers and titles": "max. 45 characters incl.
    space / 3-8 words" per heading."""
    headings = re.findall(r"^#{1,6}\s+(.+)$", gen.body, re.MULTILINE)
    if not headings:
        return _result(
            kpi,
            value=DeviationYesNo.no.value,
            indicator="DeviationYesNo",
            passed=True,
            reason="no headings in body (vacuous pass)",
            raw_metric={"headings": 0},
        )
    bad = [
        h
        for h in headings
        if len(h.strip()) > 45 or not (3 <= len(re.findall(r"\b\w+\b", h)) <= 8)
    ]
    passed = not bad
    return _result(
        kpi,
        value=(DeviationYesNo.no if passed else DeviationYesNo.yes).value,
        indicator="DeviationYesNo",
        passed=passed,
        reason=(
            f"all {len(headings)} heading(s) within 45 chars / 3-8 words"
            if passed
            else f"{len(bad)}/{len(headings)} heading(s) outside 45 chars / 3-8 words"
        ),
        raw_metric={"total": len(headings), "out_of_norm": bad[:5]},
    )


def check_keyword_density(kpi: KPI, req: EvalRequest, gen: ContentResult) -> KPIResult:
    """Workbook 07.05.01 "Body content - key word density": "min. 1 keyword
    for 100-199 words; 2 keywords for 200-1499 words; 3 keywords for
    ≥ 1500 words"."""
    keywords = _intent_keywords(req)
    if keywords is None:
        return _skipped(
            kpi, indicator="DeviationYesNo",
            reason="no intent keywords supplied; density check skipped",
        )
    body = _strip_md(gen.body).lower()
    words = _word_count(gen.body)
    required = 1 if words < 200 else 2 if words < 1500 else 3
    count = sum(body.count(k) for k in keywords)
    passed = count >= required
    return _result(
        kpi,
        value=(DeviationYesNo.no if passed else DeviationYesNo.yes).value,
        indicator="DeviationYesNo",
        passed=passed,
        reason=f"{count} keyword occurrence(s) in {words} words (norm: min {required})",
        raw_metric={"count": count, "words": words, "required": required},
    )


def check_main_keyword_first_words(
    kpi: KPI, req: EvalRequest, gen: ContentResult, *, window: int = 50
) -> KPIResult:
    """Workbook 07.05.02 "Body content - key words in first words": "min. 1
    main keyword" — Textmetrics checks the first 50 words."""
    keywords = _intent_keywords(req)
    if keywords is None:
        return _skipped(
            kpi, indicator="DeviationYesNo",
            reason="no intent keywords supplied; first-words check skipped",
        )
    opening = " ".join(re.findall(r"\b[\w']+\b", _strip_md(gen.body))[:window]).lower()
    hit = next((k for k in keywords if k in opening), None)
    passed = bool(hit)
    return _result(
        kpi,
        value=(DeviationYesNo.no if passed else DeviationYesNo.yes).value,
        indicator="DeviationYesNo",
        passed=passed,
        reason=(
            f"keyword {hit!r} in first {window} words"
            if hit
            else f"no topic keyword in first {window} words"
        ),
        raw_metric={"keywords": keywords, "window": window},
    )


# ---------- dispatcher ----------------------------------------------------

CheckFn = Callable[[KPI, EvalRequest, ContentResult], KPIResult]

# Map ``kpi.id`` (the slug from the catalogue) → default generation-time
# checker. KPIs missing from this registry stay out of the active evaluation
# stage; they can still be wired into a stricter audit profile later.
CHECK_REGISTRY: dict[str, CheckFn] = {
    # ── Objective content defects ──────────────────────────────────────
    "factuality_truthfullness": check_factuality_no_hallucinated_citations,
    "images_with_missing_alt_text": check_images_alt_present,
    # ── Source governance ──────────────────────────────────────────────
    "tracability": check_tracability,
    "approved_source_content_for_genai": check_approved_source_for_genai,
}


def run_tier1(
    *,
    catalogue: Catalogue,
    req: EvalRequest,
    gen: ContentResult,
    origin: Origin,
    channel: Channel,
) -> list[KPIResult]:
    """Run registered Tier 1 checks that apply to this content.

    The workbook marks source-management rows as "Only applicable for GenAI
    source" on instantly generated output. We honour that default for
    tracability so an instant draft is not rejected only because the writer
    omitted source IDs. If a draft does carry citations, the explicit
    approved-source exclusion check is still added because that metadata is
    objective and generation-critical.
    """
    applicable = list(catalogue.applicable(origin=origin, channel=channel))
    if gen.citations and "approved_source_content_for_genai" in CHECK_REGISTRY:
        try:
            approved_source = catalogue.by_id("approved_source_content_for_genai")
        except KeyError:
            approved_source = None
        if approved_source is not None and all(
            k.id != approved_source.id for k in applicable
        ):
            applicable.append(approved_source)
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
