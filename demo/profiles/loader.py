"""Profile loader for AURORA.

Profiles are YAML files under `profiles/workflow/` and `profiles/domain_expert/`.
Files whose name starts with `_` are ignored. The loader implements the
activation logic described in `docs/profiles.md` §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Union

import yaml

PROFILES_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = PROFILES_DIR / "workflow"
DOMAIN_EXPERT_DIR = PROFILES_DIR / "domain_expert"

CANONICAL_INTENT_CODES = frozenset(
    {"T1_DRAFT", "T1_TRANSLATE", "T1_SEARCH", "T2_COMPLIANCE", "T4_RENEWAL"}
)


@dataclass(frozen=True)
class WorkflowProfile:
    id: str
    name: str
    description: str
    activates_on_intent_codes: tuple[str, ...]
    knowledge: tuple[str, ...]
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    guardrails: tuple[str, ...]
    outputs: tuple[str, ...]
    co_activates_with: tuple[str, ...]
    category: str = "workflow"


@dataclass(frozen=True)
class DomainExpertProfile:
    id: str
    name: str
    description: str
    sector: str
    topic_keywords: tuple[str, ...]
    knowledge: tuple[str, ...]
    expertise_areas: tuple[str, ...]
    style_signature: tuple[str, ...]
    co_activates_with: tuple[str, ...]
    category: str = "domain_expert"


Profile = Union[WorkflowProfile, DomainExpertProfile]


@dataclass(frozen=True)
class ProfileBundle:
    workflow: tuple[WorkflowProfile, ...] = field(default_factory=tuple)
    domain_expert: tuple[DomainExpertProfile, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterable[Profile]:
        yield from self.workflow
        yield from self.domain_expert

    def is_empty(self) -> bool:
        return not self.workflow and not self.domain_expert


# ── Parsing ───────────────────────────────────────────────────────────────────


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at top level, got {type(data).__name__}")
    return data


def _require(d: dict, key: str, path: Path):
    if key not in d:
        raise ValueError(f"{path}: missing required field '{key}'")
    return d[key]


def _tuple(d: dict, key: str, path: Path, *, required: bool = True) -> tuple[str, ...]:
    if key not in d:
        if required:
            raise ValueError(f"{path}: missing required field '{key}'")
        return ()
    val = d[key]
    if val is None:
        return ()
    if not isinstance(val, list):
        raise ValueError(f"{path}: '{key}' must be a list")
    return tuple(str(x) for x in val)


def _parse_workflow(d: dict, path: Path) -> WorkflowProfile:
    activates_on = _require(d, "activates_on", path) or {}
    capabilities = _require(d, "capabilities", path) or {}
    intent_codes = activates_on.get("intent_codes") or []
    skills = capabilities.get("skills") or []
    return WorkflowProfile(
        id=_require(d, "id", path),
        name=_require(d, "name", path),
        description=_require(d, "description", path),
        activates_on_intent_codes=tuple(intent_codes),
        knowledge=_tuple(d, "knowledge", path),
        skills=tuple(skills),
        tools=_tuple(d, "tools", path, required=False),
        guardrails=_tuple(d, "guardrails", path, required=False),
        outputs=_tuple(d, "outputs", path, required=False),
        co_activates_with=_tuple(d, "co_activates_with", path),
    )


def _parse_expert(d: dict, path: Path) -> DomainExpertProfile:
    activates_on = _require(d, "activates_on", path) or {}
    capabilities = _require(d, "capabilities", path) or {}
    if "sector" not in activates_on:
        raise ValueError(f"{path}: domain_expert profiles require activates_on.sector")
    return DomainExpertProfile(
        id=_require(d, "id", path),
        name=_require(d, "name", path),
        description=_require(d, "description", path),
        sector=str(activates_on["sector"]),
        topic_keywords=tuple(activates_on.get("topic_keywords") or []),
        knowledge=_tuple(d, "knowledge", path),
        expertise_areas=tuple(capabilities.get("expertise_areas") or []),
        style_signature=_tuple(d, "style_signature", path, required=False),
        co_activates_with=_tuple(d, "co_activates_with", path),
    )


def _parse(path: Path) -> Profile:
    d = _read_yaml(path)
    category = _require(d, "category", path)
    if category == "workflow":
        return _parse_workflow(d, path)
    if category == "domain_expert":
        return _parse_expert(d, path)
    raise ValueError(f"{path}: unknown category '{category}'")


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".yaml" and not p.name.startswith("_"))


# ── Public API ────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_all() -> ProfileBundle:
    """Load every profile from disk. Cached for the process lifetime."""
    workflow = tuple(_parse(p) for p in _yaml_files(WORKFLOW_DIR))
    expert = tuple(_parse(p) for p in _yaml_files(DOMAIN_EXPERT_DIR))
    return ProfileBundle(workflow=workflow, domain_expert=expert)


def load_by_id(profile_id: str) -> Profile:
    bundle = load_all()
    for p in bundle:
        if p.id == profile_id:
            return p
    raise KeyError(f"profile id '{profile_id}' not found")


def match(
    intent_code: str | None = None,
    sector: str | None = None,
    keywords: Iterable[str] = (),
) -> ProfileBundle:
    """Return profiles activated by the given intent / sector / keywords.

    See docs/profiles.md §3. A request typically activates one workflow profile
    (matched by intent_code) and zero-or-more domain experts (matched by sector
    AND any keyword overlap). Matching is case-insensitive on keywords.
    """
    bundle = load_all()
    keyword_set = {k.lower() for k in keywords}

    workflow_matches = (
        tuple(w for w in bundle.workflow if intent_code in w.activates_on_intent_codes)
        if intent_code is not None
        else ()
    )

    expert_matches: tuple[DomainExpertProfile, ...] = ()
    if sector is not None:
        expert_matches = tuple(
            e
            for e in bundle.domain_expert
            if e.sector == sector
            and (not keyword_set or keyword_set.intersection({k.lower() for k in e.topic_keywords}))
        )

    return ProfileBundle(workflow=workflow_matches, domain_expert=expert_matches)
