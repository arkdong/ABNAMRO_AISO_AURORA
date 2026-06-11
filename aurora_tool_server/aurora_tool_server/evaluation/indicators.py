"""Indicator enums used as judge / check outputs.

Every leaf KPI in the workbook has a defined indicator scale (col 18 of the
`Inventory` sheet). The build script (``rag/scripts/build_kpi_catalogue.py``)
maps each free-text phrase to one of the enum classes below; consumers
(``tier1_deterministic`` and ``tier2_judges``) emit *values* of those enums so
results merge into the existing PowerBI dashboards in the workbook's
vocabulary unchanged.

A judge result that fails to produce a valid enum value falls back to the
sentinel ``"unknown"`` value (defined per-scale) — the service tier then
flags it as ``passed=False`` with the reason ``"judge error"``.
"""

from __future__ import annotations

from enum import Enum


class Maturity(str, Enum):
    """Category-level audit rollups (``low, medium or high maturity``)."""

    low = "low"
    medium = "medium"
    high = "high"


class PresenceScale(str, Enum):
    """``present, not present`` — most-used scale in the workbook."""

    present = "present"
    not_present = "not_present"
    unknown = "unknown"


class YesNoScale(str, Enum):
    """``completed step yes/no`` — used for dCLP editorial steps."""

    yes = "yes"
    no = "no"
    unknown = "unknown"


class DeviationYesNo(str, Enum):
    """``yes/no deviation from norm``."""

    yes = "yes"
    no = "no"
    unknown = "unknown"


class DeviationScale(str, Enum):
    """``many deviations, few deviations, no deviations``."""

    many = "many"
    few = "few"
    none_ = "none"
    unknown = "unknown"


class AmbiguityScale(str, Enum):
    """``many ambiguities, few ambiguities, no ambiguities`` (Clarity KPI)."""

    many = "many"
    few = "few"
    none_ = "none"
    unknown = "unknown"


class RelevanceScale(str, Enum):
    """``off-topic, somewhat relevant, reasonable relevant, highly relevant``."""

    off_topic = "off_topic"
    somewhat = "somewhat"
    reasonable = "reasonable"
    highly = "highly"
    unknown = "unknown"


class GroundednessScale(str, Enum):
    """``no grounding, limited grounding, reasonable grounding, fully grounded``."""

    none_ = "none"
    limited = "limited"
    reasonable = "reasonable"
    full = "full"
    unknown = "unknown"


class ErrorScale(str, Enum):
    """``numerous, several, moderate, few, no errors`` (Factuality KPI)."""

    numerous = "numerous"
    several = "several"
    moderate = "moderate"
    few = "few"
    none_ = "none"
    unknown = "unknown"


class CompletenessScale(str, Enum):
    """``very incomplete, incomplete, fairly complete, mostly complete, fully complete``."""

    very_incomplete = "very_incomplete"
    incomplete = "incomplete"
    fairly = "fairly"
    mostly = "mostly"
    full = "full"
    unknown = "unknown"


class ClarityScale(str, Enum):
    """``unclear, somewhat clear, clear, very clear``."""

    unclear = "unclear"
    somewhat = "somewhat"
    clear = "clear"
    very_clear = "very_clear"
    unknown = "unknown"


class FitScale(str, Enum):
    """``no fit, limited fit, optimal fit``."""

    none_ = "none"
    limited = "limited"
    optimal = "optimal"
    unknown = "unknown"


class OptionsScale(str, Enum):
    """``no options, limited options, many options``."""

    none_ = "none"
    limited = "limited"
    many = "many"
    unknown = "unknown"


class FivePointScale(str, Enum):
    """``very low, low, medium, high, very high``."""

    very_low = "very_low"
    low = "low"
    medium = "medium"
    high = "high"
    very_high = "very_high"
    unknown = "unknown"


class LengthScale(str, Enum):
    """``right length, too long``."""

    right = "right"
    too_long = "too_long"
    unknown = "unknown"


class LanguageLevelScale(str, Enum):
    """``A1, A2, B1, B2, C1, C2``."""

    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"
    unknown = "unknown"


class UsedScale(str, Enum):
    """``used, not used`` (e.g. source ID + version tag)."""

    used = "used"
    not_used = "not_used"
    unknown = "unknown"


class ApplicableScale(str, Enum):
    """``applicable, not applicable``."""

    applicable = "applicable"
    not_applicable = "not_applicable"
    unknown = "unknown"


class ExclusionScale(str, Enum):
    """``exclusion. no exclusion`` (GenAI source exclusion tag)."""

    exclusion = "exclusion"
    no_exclusion = "no_exclusion"
    unknown = "unknown"


# Engagement scales — currently out of scope for generation-time eval, kept
# so the catalogue mapping is complete.

class CESScale(str, Enum):
    """``1 very difficult, 2 difficult, 3 neutral, 4 easy, 5 very easy``."""

    very_difficult = "1_very_difficult"
    difficult = "2_difficult"
    neutral = "3_neutral"
    easy = "4_easy"
    very_easy = "5_very_easy"
    unknown = "unknown"


class CSATScale(str, Enum):
    very_dissatisfied = "1_very_dissatisfied"
    dissatisfied = "2_dissatisfied"
    neutral = "3_neutral"
    satisfied = "4_satisfied"
    very_satisfied = "5_very_satisfied"
    unknown = "unknown"


class NPSScale(str, Enum):
    detractor = "detractor"
    passive = "passive"
    promoter = "promoter"
    unknown = "unknown"


class PublishedScale(str, Enum):
    published = "published"
    not_published = "not_published"
    unknown = "unknown"


class SentimentScale(str, Enum):
    neutral = "neutral"
    positive = "positive"
    negative = "negative"
    unknown = "unknown"


class GenderScale(str, Enum):
    neutral = "neutral"
    male = "male"
    female = "female"
    unknown = "unknown"


# Registry — string name → enum class. The catalogue stores indicator names
# as strings, so consumers look up the class via this map.
INDICATOR_REGISTRY: dict[str, type[Enum]] = {
    "Maturity": Maturity,
    "PresenceScale": PresenceScale,
    "YesNoScale": YesNoScale,
    "DeviationYesNo": DeviationYesNo,
    "DeviationScale": DeviationScale,
    "AmbiguityScale": AmbiguityScale,
    "RelevanceScale": RelevanceScale,
    "GroundednessScale": GroundednessScale,
    "ErrorScale": ErrorScale,
    "CompletenessScale": CompletenessScale,
    "ClarityScale": ClarityScale,
    "FitScale": FitScale,
    "OptionsScale": OptionsScale,
    "FivePointScale": FivePointScale,
    "LengthScale": LengthScale,
    "LanguageLevelScale": LanguageLevelScale,
    "UsedScale": UsedScale,
    "ApplicableScale": ApplicableScale,
    "ExclusionScale": ExclusionScale,
    "CESScale": CESScale,
    "CSATScale": CSATScale,
    "NPSScale": NPSScale,
    "PublishedScale": PublishedScale,
    "SentimentScale": SentimentScale,
    "GenderScale": GenderScale,
}


# Per-scale: which enum values count as acceptable for the default
# generation-time review. This is intentionally softer than a final
# publication audit: minor issues stay visible in the KPI value/reason, but
# they do not make an otherwise usable generation fail the stage.
PASSING_VALUES: dict[type[Enum], set[Enum]] = {
    PresenceScale: {PresenceScale.present},
    YesNoScale: {YesNoScale.yes},
    DeviationYesNo: {DeviationYesNo.no},
    DeviationScale: {DeviationScale.few, DeviationScale.none_},
    AmbiguityScale: {AmbiguityScale.few, AmbiguityScale.none_},
    RelevanceScale: {
        RelevanceScale.somewhat,
        RelevanceScale.reasonable,
        RelevanceScale.highly,
    },
    GroundednessScale: {GroundednessScale.reasonable, GroundednessScale.full},
    ErrorScale: {ErrorScale.few, ErrorScale.none_},
    CompletenessScale: {
        CompletenessScale.fairly,
        CompletenessScale.mostly,
        CompletenessScale.full,
    },
    ClarityScale: {ClarityScale.clear, ClarityScale.very_clear},
    FitScale: {FitScale.optimal},
    OptionsScale: {OptionsScale.limited, OptionsScale.many},
    FivePointScale: {FivePointScale.high, FivePointScale.very_high},
    LengthScale: {LengthScale.right},
    UsedScale: {UsedScale.used},
    ApplicableScale: {ApplicableScale.applicable},
    ExclusionScale: {ExclusionScale.no_exclusion},
    Maturity: {Maturity.medium, Maturity.high},
    PublishedScale: {PublishedScale.published},
    LanguageLevelScale: {
        LanguageLevelScale.A1,
        LanguageLevelScale.A2,
        LanguageLevelScale.B1,
    },
}


def is_passing(scale_cls: type[Enum], value: Enum) -> bool:
    """Default pass/fail derivation from indicator value alone."""
    passing = PASSING_VALUES.get(scale_cls)
    if passing is None:
        return False
    return value in passing
