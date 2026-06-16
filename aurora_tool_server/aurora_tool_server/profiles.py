"""Standalone profile loading and matching."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import PROFILES_DIR
from .schemas import (
    IntentResult,
    ProfileBundleResult,
    ProfileCategory,
    ProfileResult,
    ProfileWriteRequest,
)

WORKFLOW_DIR = PROFILES_DIR / "workflow"
DOMAIN_EXPERT_DIR = PROFILES_DIR / "domain_expert"


class ProfileStoreError(RuntimeError):
    """Base error raised for profile catalogue mutations."""


class ProfileNotFoundError(ProfileStoreError):
    """Raised when a profile id cannot be found in the requested category."""


class ProfileConflictError(ProfileStoreError):
    """Raised when creating a profile would duplicate an existing id."""


class ProfileValidationError(ProfileStoreError):
    """Raised when a profile mutation request is internally inconsistent."""


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.suffix == ".yaml" and not p.name.startswith("_")
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _category_dir(category: ProfileCategory) -> Path:
    if category == "workflow":
        return WORKFLOW_DIR
    return DOMAIN_EXPERT_DIR


def _find_profile_path(category: ProfileCategory, profile_id: str) -> Path | None:
    for path in _yaml_files(_category_dir(category)):
        data = _read_yaml(path)
        if str(data.get("id") or "") == profile_id:
            return path
    return None


def _profile_id_exists(profile_id: str) -> bool:
    return any(profile.id == profile_id for profile in load_profiles().all_profiles)


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _parse_profile(path: Path) -> ProfileResult:
    data = _read_yaml(path)
    category = data.get("category")
    activates_on = data.get("activates_on") or {}
    capabilities = data.get("capabilities") or {}

    if category == "workflow":
        return ProfileResult(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            category="workflow",
            activates_on_intent_codes=_list(activates_on.get("intent_codes")),
            knowledge=_list(data.get("knowledge")),
            skills=_list(capabilities.get("skills")),
            tools=_list(data.get("tools")),
            guardrails=_list(data.get("guardrails")),
            outputs=_list(data.get("outputs")),
            co_activates_with=_list(data.get("co_activates_with")),
        )

    if category == "domain_expert":
        return ProfileResult(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            category="domain_expert",
            sector=str(activates_on.get("sector") or ""),
            topic_keywords=_list(activates_on.get("topic_keywords")),
            knowledge=_list(data.get("knowledge")),
            expertise_areas=_list(capabilities.get("expertise_areas")),
            style_signature=_list(data.get("style_signature")),
            co_activates_with=_list(data.get("co_activates_with")),
        )

    raise ValueError(f"{path}: unknown profile category {category!r}")


@lru_cache(maxsize=1)
def load_profiles() -> ProfileBundleResult:
    return ProfileBundleResult(
        workflow=[_parse_profile(path) for path in _yaml_files(WORKFLOW_DIR)],
        domain_expert=[_parse_profile(path) for path in _yaml_files(DOMAIN_EXPERT_DIR)],
    )


def list_profiles() -> ProfileBundleResult:
    """Return the editable profile catalogue."""
    return load_profiles()


def _profile_to_yaml(profile: ProfileWriteRequest) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "category": profile.category,
    }
    if profile.category == "workflow":
        data.update(
            {
                "activates_on": {
                    "intent_codes": profile.activates_on_intent_codes,
                },
                "knowledge": profile.knowledge,
                "capabilities": {
                    "skills": profile.skills,
                },
                "tools": profile.tools,
                "guardrails": profile.guardrails,
                "outputs": profile.outputs,
                "co_activates_with": profile.co_activates_with,
            }
        )
        return data

    data.update(
        {
            "activates_on": {
                "sector": profile.sector or "",
                "topic_keywords": profile.topic_keywords,
            },
            "knowledge": profile.knowledge,
            "capabilities": {
                "expertise_areas": profile.expertise_areas,
            },
            "style_signature": profile.style_signature,
            "co_activates_with": profile.co_activates_with,
        }
    )
    return data


def _write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    tmp_path.replace(path)
    load_profiles.cache_clear()


def create_profile(profile: ProfileWriteRequest) -> ProfileResult:
    """Create a new YAML-backed profile and return the parsed result."""
    if _profile_id_exists(profile.id):
        raise ProfileConflictError(f"profile id {profile.id!r} already exists")

    path = _category_dir(profile.category) / f"{profile.id}.yaml"
    if path.exists():
        raise ProfileConflictError(f"profile file {path.name!r} already exists")
    _write_yaml_atomic(path, _profile_to_yaml(profile))
    return _parse_profile(path)


def update_profile(
    category: ProfileCategory,
    profile_id: str,
    profile: ProfileWriteRequest,
) -> ProfileResult:
    """Update an existing profile while preserving its current YAML filename."""
    if profile.category != category or profile.id != profile_id:
        raise ProfileValidationError("profile id and category cannot be changed")

    path = _find_profile_path(category, profile_id)
    if path is None:
        raise ProfileNotFoundError(f"profile {category}/{profile_id} was not found")

    _write_yaml_atomic(path, _profile_to_yaml(profile))
    return _parse_profile(path)


def delete_profile(category: ProfileCategory, profile_id: str) -> ProfileResult:
    """Delete a YAML-backed profile and return the deleted profile payload."""
    path = _find_profile_path(category, profile_id)
    if path is None:
        raise ProfileNotFoundError(f"profile {category}/{profile_id} was not found")

    deleted = _parse_profile(path)
    path.unlink()
    load_profiles.cache_clear()
    return deleted


def _deterministic_select(intent: IntentResult, *, reasoning: str = "") -> ProfileBundleResult:
    all_profiles = load_profiles()
    task_codes = set(intent.task_codes)
    keyword_set = {kw.lower() for kw in intent.topic_keywords}

    workflow = [
        profile
        for profile in all_profiles.workflow
        if task_codes.intersection(set(profile.activates_on_intent_codes))
    ]

    experts: list[ProfileResult] = []
    if intent.sector:
        for profile in all_profiles.domain_expert:
            if profile.sector != intent.sector:
                continue
            profile_keywords = {kw.lower() for kw in profile.topic_keywords}
            if not keyword_set or keyword_set.intersection(profile_keywords):
                experts.append(profile)

    return ProfileBundleResult(
        workflow=workflow,
        domain_expert=experts,
        source="deterministic",
        reasoning=reasoning or "Matched workflow intent codes and domain profile sector/keywords.",
    )


def _profile_catalogue(bundle: ProfileBundleResult) -> str:
    entries: list[dict[str, Any]] = []
    for profile in bundle.workflow:
        entries.append(
            {
                "id": profile.id,
                "name": profile.name,
                "category": profile.category,
                "description": profile.description,
                "activates_on_intent_codes": profile.activates_on_intent_codes,
                "skills": profile.skills,
                "outputs": profile.outputs,
                "guardrails": profile.guardrails,
            }
        )
    for profile in bundle.domain_expert:
        entries.append(
            {
                "id": profile.id,
                "name": profile.name,
                "category": profile.category,
                "description": profile.description,
                "sector": profile.sector,
                "topic_keywords": profile.topic_keywords,
                "expertise_areas": profile.expertise_areas,
                "style_signature": profile.style_signature,
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    return json.loads(content)


_SYSTEM_PROMPT = """You select AURORA workflow and domain-expert profiles.

Return JSON with:
- workflow_ids: list of workflow profile ids to activate
- domain_expert_ids: list of domain_expert profile ids to activate
- reasons: object mapping selected profile id to a short reason
- reasoning: one short sentence summarizing the selection

Rules:
- Only select ids present in the provided profile catalogue.
- Select workflow profiles that directly support the classified task_codes.
- Select domain experts whose sector and topic vocabulary match the request.
- Prefer a focused set over selecting everything.
- If uncertain, preserve the obvious workflow profile for the primary task.
"""


def _clone_with_reason(profile: ProfileResult, reason: str) -> ProfileResult:
    return profile.model_copy(update={"selection_reason": reason})


def select_profiles(
    intent: IntentResult,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> ProfileBundleResult:
    fallback = _deterministic_select(intent)
    if not api_key or not model:
        return fallback

    all_profiles = load_profiles()
    workflow_by_id = {profile.id: profile for profile in all_profiles.workflow}
    expert_by_id = {profile.id: profile for profile in all_profiles.domain_expert}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Intent:\n"
                        f"{intent.model_dump_json(indent=2)}\n\n"
                        "Profile catalogue:\n"
                        f"{_profile_catalogue(all_profiles)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = _parse_json(completion.choices[0].message.content or "{}")
        reasons = parsed.get("reasons") if isinstance(parsed.get("reasons"), dict) else {}
        workflow = [
            _clone_with_reason(workflow_by_id[profile_id], str(reasons.get(profile_id) or "Selected by LLM."))
            for profile_id in parsed.get("workflow_ids", [])
            if profile_id in workflow_by_id
        ]
        experts = [
            _clone_with_reason(expert_by_id[profile_id], str(reasons.get(profile_id) or "Selected by LLM."))
            for profile_id in parsed.get("domain_expert_ids", [])
            if profile_id in expert_by_id
        ]
        if not workflow and not experts:
            return _deterministic_select(
                intent,
                reasoning="LLM profile selector returned no usable ids; deterministic fallback.",
            )
        return ProfileBundleResult(
            workflow=workflow,
            domain_expert=experts,
            source="llm",
            reasoning=str(parsed.get("reasoning") or "Profiles selected by configured LLM."),
        )
    except Exception as exc:
        return _deterministic_select(
            intent,
            reasoning=f"LLM profile selector failed; deterministic fallback: {exc}",
        )
