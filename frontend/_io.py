"""Profile YAML write helpers for the Streamlit UI.

Reads go through ``profiles.loader`` (the canonical path). Writes assemble the
nested shape the loader expects, dump via PyYAML, then clear the loader cache
and run :func:`profiles.validate.validate` so the UI can surface any issues.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiles import load_all  # noqa: E402
from profiles.loader import DOMAIN_EXPERT_DIR, WORKFLOW_DIR  # noqa: E402
from profiles.validate import validate  # noqa: E402


def category_dir(category: str) -> Path:
    if category == "workflow":
        return WORKFLOW_DIR
    if category == "domain_expert":
        return DOMAIN_EXPERT_DIR
    raise ValueError(f"unknown category: {category}")


def profile_path(profile_id: str, category: str) -> Path:
    stem = profile_id[len("expert_"):] if profile_id.startswith("expert_") else profile_id
    return category_dir(category) / f"{stem}.yaml"


def build_yaml_dict(form_data: dict, category: str) -> dict[str, Any]:
    """Assemble the nested loader-expected shape from flat form data."""
    activates_on: dict[str, Any] = {}
    capabilities: dict[str, Any] = {}

    out: dict[str, Any] = {
        "id": form_data["id"],
        "name": form_data["name"],
        "description": form_data["description"],
        "category": category,
    }

    if category == "workflow":
        activates_on["intent_codes"] = list(form_data.get("intent_codes", []))
        capabilities["skills"] = list(form_data.get("skills", []))
    else:
        activates_on["sector"] = form_data.get("sector", "")
        activates_on["topic_keywords"] = list(form_data.get("topic_keywords", []))
        capabilities["expertise_areas"] = list(form_data.get("expertise_areas", []))

    out["activates_on"] = activates_on
    out["knowledge"] = list(form_data.get("knowledge", []))
    out["capabilities"] = capabilities

    if category == "workflow":
        if form_data.get("tools"):
            out["tools"] = list(form_data["tools"])
        if form_data.get("guardrails"):
            out["guardrails"] = list(form_data["guardrails"])
        if form_data.get("outputs"):
            out["outputs"] = list(form_data["outputs"])
    else:
        if form_data.get("style_signature"):
            out["style_signature"] = list(form_data["style_signature"])

    out["co_activates_with"] = list(form_data.get("co_activates_with", []))
    return out


def save_profile(form_data: dict, category: str) -> tuple[Path, list[str]]:
    """Write profile YAML and run validation. Returns (path, errors)."""
    data = build_yaml_dict(form_data, category)
    path = profile_path(data["id"], category)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    load_all.cache_clear()
    return path, validate()


def delete_profile(profile_id: str, category: str) -> tuple[Path, list[str]]:
    path = profile_path(profile_id, category)
    if path.exists():
        path.unlink()
    load_all.cache_clear()
    return path, validate()
