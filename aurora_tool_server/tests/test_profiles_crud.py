from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aurora_tool_server import profiles as profile_store
from aurora_tool_server.schemas import ProfileWriteRequest


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.fixture
def temp_profile_store(tmp_path, monkeypatch):
    workflow_dir = tmp_path / "profiles" / "workflow"
    domain_dir = tmp_path / "profiles" / "domain_expert"
    _write_yaml(
        workflow_dir / "legacy_workflow_filename.yaml",
        {
            "id": "existing_workflow",
            "name": "Existing Workflow",
            "description": "Original workflow description.",
            "category": "workflow",
            "activates_on": {"intent_codes": ["T1_DRAFT"]},
            "knowledge": ["Original workflow knowledge"],
            "capabilities": {"skills": ["Original skill"]},
            "tools": [],
            "guardrails": [],
            "outputs": [],
            "co_activates_with": [],
        },
    )
    _write_yaml(
        domain_dir / "legacy_domain_filename.yaml",
        {
            "id": "existing_domain",
            "name": "Existing Domain",
            "description": "Original domain description.",
            "category": "domain_expert",
            "activates_on": {
                "sector": "Technologie, Media & Telecom",
                "topic_keywords": ["AI"],
            },
            "knowledge": ["Original domain knowledge"],
            "capabilities": {"expertise_areas": ["Original expertise"]},
            "style_signature": ["Original style"],
            "co_activates_with": [],
        },
    )
    monkeypatch.setattr(profile_store, "WORKFLOW_DIR", workflow_dir)
    monkeypatch.setattr(profile_store, "DOMAIN_EXPERT_DIR", domain_dir)
    profile_store.load_profiles.cache_clear()
    yield workflow_dir, domain_dir
    profile_store.load_profiles.cache_clear()


def test_create_update_and_delete_workflow_profile_refreshes_cache(temp_profile_store):
    workflow_dir, _domain_dir = temp_profile_store
    cached = profile_store.list_profiles()
    assert cached.workflow[0].description == "Original workflow description."

    created = profile_store.create_profile(
        ProfileWriteRequest(
            id="new_workflow",
            name="New Workflow",
            description="New workflow description.",
            category="workflow",
            activates_on_intent_codes=["T1_SEARCH"],
            knowledge=["New knowledge"],
            skills=["New skill"],
        )
    )

    assert created.id == "new_workflow"
    assert (workflow_dir / "new_workflow.yaml").is_file()

    updated = profile_store.update_profile(
        "workflow",
        "existing_workflow",
        ProfileWriteRequest(
            id="existing_workflow",
            name="Updated Workflow",
            description="Updated workflow description.",
            category="workflow",
            activates_on_intent_codes=["T2_COMPLIANCE"],
            knowledge=["Updated knowledge"],
            skills=["Updated skill"],
            guardrails=["Updated guardrail"],
        ),
    )

    assert updated.name == "Updated Workflow"
    assert (workflow_dir / "legacy_workflow_filename.yaml").is_file()
    assert not (workflow_dir / "existing_workflow.yaml").exists()
    assert profile_store.list_profiles().workflow[0].description == (
        "Updated workflow description."
    )

    deleted = profile_store.delete_profile("workflow", "new_workflow")
    remaining_ids = {profile.id for profile in profile_store.list_profiles().workflow}

    assert deleted.id == "new_workflow"
    assert "new_workflow" not in remaining_ids
    assert not (workflow_dir / "new_workflow.yaml").exists()


def test_create_domain_expert_profile_round_trips(temp_profile_store):
    _workflow_dir, domain_dir = temp_profile_store

    created = profile_store.create_profile(
        ProfileWriteRequest(
            id="new_domain",
            name="New Domain",
            description="New domain description.",
            category="domain_expert",
            sector="Technologie, Media & Telecom",
            topic_keywords=["cybersecurity", "AI"],
            knowledge=["Domain knowledge"],
            expertise_areas=["Risk analysis"],
            style_signature=["Analytical"],
            co_activates_with=["drafter"],
        )
    )

    assert created.id == "new_domain"
    assert created.sector == "Technologie, Media & Telecom"
    assert created.topic_keywords == ["cybersecurity", "AI"]
    assert (domain_dir / "new_domain.yaml").is_file()


def test_duplicate_profile_id_is_rejected(temp_profile_store):
    with pytest.raises(profile_store.ProfileConflictError):
        profile_store.create_profile(
            ProfileWriteRequest(
                id="existing_domain",
                name="Duplicate",
                description="Duplicate id.",
                category="workflow",
            )
        )


def test_create_rejects_existing_yaml_filename(temp_profile_store):
    with pytest.raises(profile_store.ProfileConflictError):
        profile_store.create_profile(
            ProfileWriteRequest(
                id="legacy_domain_filename",
                name="Filename Collision",
                description="Would overwrite an existing file.",
                category="domain_expert",
            )
        )


def test_update_rejects_identity_changes(temp_profile_store):
    with pytest.raises(profile_store.ProfileValidationError):
        profile_store.update_profile(
            "workflow",
            "existing_workflow",
            ProfileWriteRequest(
                id="renamed_workflow",
                name="Renamed Workflow",
                description="Should not be accepted.",
                category="workflow",
            ),
        )


def test_missing_profile_delete_raises_not_found(temp_profile_store):
    with pytest.raises(profile_store.ProfileNotFoundError):
        profile_store.delete_profile("domain_expert", "missing_domain")


def test_profile_write_request_rejects_invalid_ids():
    with pytest.raises(ValidationError):
        ProfileWriteRequest(
            id="Bad ID",
            name="Bad",
            description="Invalid id.",
            category="workflow",
        )
