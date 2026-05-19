"""KPI catalogue loader.

Reads ``backend/evaluation/data/kpi_catalogue.json`` (built by
``rag/scripts/build_kpi_catalogue.py``) and exposes typed accessors. Loaded
once via ``lru_cache``; the file is small (~130 entries) so reload cost is
negligible if a test calls ``load_catalogue.cache_clear()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from backend.evaluation.types import Channel, Origin

CATALOGUE_PATH = Path(__file__).resolve().parent / "data" / "kpi_catalogue.json"


@dataclass(frozen=True)
class KPI:
    id: str
    name: str
    category: Optional[str]
    cluster_short: Optional[str]
    primary_cluster: Optional[str]
    secondary_cluster: Optional[str]
    monitoring: str  # "Mandatory" | "Optional"
    weight: str  # "Blocking" | "High" | "Medium" | "Low"
    contribution: Optional[str]
    norm: Optional[str]
    norm_chat: Optional[str]
    norm_messages: Optional[str]
    measurement: Optional[str]
    indicator: Optional[str]  # enum class name, e.g. "ErrorScale"
    indicator_phrase: Optional[str]
    automated_match: Optional[str]
    relevance: dict[str, Optional[str]] = field(default_factory=dict)
    guardrail_category: Optional[str] = None
    guardrail_tag: Optional[str] = None

    @property
    def is_blocking(self) -> bool:
        return self.weight == "Blocking" and self.monitoring == "Mandatory"

    def norm_for(self, channel: Channel) -> Optional[str]:
        """Channel-specific norm override, falling back to the generic norm."""
        if channel == "chat" and self.norm_chat:
            return self.norm_chat
        if channel == "messages" and self.norm_messages:
            return self.norm_messages
        return self.norm


@dataclass(frozen=True)
class Catalogue:
    kpis: tuple[KPI, ...]
    categories: tuple[dict, ...]
    clusters: tuple[dict, ...]

    def by_id(self, kpi_id: str) -> KPI:
        for k in self.kpis:
            if k.id == kpi_id:
                return k
        raise KeyError(kpi_id)

    def blocking(self) -> tuple[KPI, ...]:
        return tuple(k for k in self.kpis if k.is_blocking)

    def by_weight(self, weight: str) -> tuple[KPI, ...]:
        return tuple(k for k in self.kpis if k.weight == weight)

    def applicable(self, *, origin: Origin, channel: Channel) -> tuple[KPI, ...]:
        """Filter KPIs by the relevance columns in the workbook.

        A KPI applies when:
        - ``relevance[origin]`` is one of ``Applicable`` / ``Need`` /
          ``Applicable for *`` (anything that starts with ``Applicable``), AND
        - ``relevance[channel]`` is similarly ``Applicable`` (the channel
          column is only filled in for that channel's specific cell, so when
          a row leaves it blank we treat it as applicable rather than drop
          the KPI silently).
        """

        def applies(value: Optional[str]) -> bool:
            if value is None:
                return True  # missing → permissive (the workbook leaves channel cells blank for "all")
            v = value.strip().lower()
            return v.startswith("applicable") or v == "need"

        return tuple(
            k
            for k in self.kpis
            if applies(k.relevance.get(origin))
            and applies(k.relevance.get(channel))
        )

    def by_category(self) -> dict[str, list[KPI]]:
        out: dict[str, list[KPI]] = {}
        for k in self.kpis:
            cat = k.category or "Uncategorised"
            out.setdefault(cat, []).append(k)
        return out


@lru_cache(maxsize=1)
def load_catalogue(path: Optional[Path] = None) -> Catalogue:
    """Load the catalogue JSON. ``path`` is mostly for tests."""
    p = path or CATALOGUE_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    kpis = tuple(
        KPI(
            id=row["id"],
            name=row["name"],
            category=row.get("category"),
            cluster_short=row.get("cluster_short"),
            primary_cluster=row.get("primary_cluster"),
            secondary_cluster=row.get("secondary_cluster"),
            monitoring=row.get("monitoring") or "Optional",
            weight=row.get("weight") or "Medium",
            contribution=row.get("contribution"),
            norm=row.get("norm"),
            norm_chat=row.get("norm_chat"),
            norm_messages=row.get("norm_messages"),
            measurement=row.get("measurement"),
            indicator=row.get("indicator"),
            indicator_phrase=row.get("indicator_phrase"),
            automated_match=row.get("automated_match"),
            relevance=row.get("relevance") or {},
            guardrail_category=row.get("guardrail_category"),
            guardrail_tag=row.get("guardrail_tag"),
        )
        for row in raw.get("kpis", [])
    )
    return Catalogue(
        kpis=kpis,
        categories=tuple(raw.get("categories", [])),
        clusters=tuple(raw.get("clusters", [])),
    )


def kpis_by_ids(ids: Iterable[str]) -> tuple[KPI, ...]:
    cat = load_catalogue()
    out: list[KPI] = []
    for i in ids:
        try:
            out.append(cat.by_id(i))
        except KeyError:
            continue
    return tuple(out)
