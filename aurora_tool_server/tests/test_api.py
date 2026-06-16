from fastapi.testclient import TestClient
import yaml

from aurora_tool_server import AuroraConfig, AuroraCore
from aurora_tool_server import api
from aurora_tool_server import profiles as profile_store


def _client() -> TestClient:
    api.core = AuroraCore(
        AuroraConfig(
            intent_api_key=None,
            content_api_key=None,
            evaluation_api_key=None,
            retrieval_k=3,
        )
    )
    return TestClient(api.app)


def test_health_and_openapi():
    client = _client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/v1/runs" in openapi.json()["paths"]


def test_run_endpoint_and_audit_endpoint():
    client = _client()
    run_id = "run_api_pipeline"
    response = client.post(
        "/v1/runs",
        json={
            "user_prompt": (
                "Write an English article about agentic AI and cybersecurity "
                "for TMT companies."
            ),
            "run_id": run_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["run_id"] == run_id
    assert payload["retrieval"]["snippets"]
    first_citation = payload["content"]["citations"][0]
    assert first_citation["article_title"]
    assert first_citation["source_url"].startswith("https://www.abnamro.nl/")

    audit = client.get(f"/v1/runs/{payload['run_id']}/audit")
    assert audit.status_code == 200
    assert audit.json()["events"]
    assert {event["run_id"] for event in audit.json()["events"]} == {run_id}


def test_stage_endpoints_round_trip():
    client = _client()
    run_id = "run_api_stage"
    intent_response = client.post(
        "/v1/intent/classify",
        json={
            "user_prompt": "Check this retail media draft against the writing guide.",
            "run_id": run_id,
        },
    )
    assert intent_response.status_code == 200
    intent = intent_response.json()

    profiles_response = client.post(
        "/v1/profiles/select",
        json={"intent": intent, "run_id": run_id},
    )
    assert profiles_response.status_code == 200
    profiles = profiles_response.json()

    retrieval_response = client.post(
        "/v1/retrieval/search",
        json={
            "user_prompt": "Check this retail media draft against the writing guide.",
            "intent": intent,
            "profiles": profiles,
            "options": {"k": 2},
            "run_id": run_id,
        },
    )
    assert retrieval_response.status_code == 200
    retrieval = retrieval_response.json()
    assert len(retrieval["snippets"]) <= 2

    refine_response = client.post(
        "/v1/prompts/refine",
        json={
            "user_prompt": "Check this retail media draft against the writing guide.",
            "intent": intent,
            "profiles": profiles,
            "retrieval": retrieval,
            "run_id": run_id,
        },
    )
    assert refine_response.status_code == 200
    assert refine_response.json()["questions"]

    generate_response = client.post(
        "/v1/drafts/generate",
        json={
            "refined_prompt": "Draft it",
            "intent": intent,
            "profiles": profiles,
            "snippets": [
                {
                    "source_doc": "corpus_en",
                    "node_id": "n1",
                    "title": "Matched section",
                    "article_title": "Linked article title",
                    "source_url": "https://www.abnamro.nl/example.html",
                    "content": "Grounding text.",
                    "score": 1.0,
                    "reason": "test",
                }
            ],
            "run_id": run_id,
        },
    )
    assert generate_response.status_code == 200
    citation = generate_response.json()["citations"][0]
    assert citation["article_title"] == "Linked article title"
    assert citation["source_url"] == "https://www.abnamro.nl/example.html"

    audit = client.get(f"/v1/runs/{run_id}/audit")
    assert audit.status_code == 200
    assert [event["stage"] for event in audit.json()["events"]] == [
        "intent",
        "profiles",
        "retrieval",
        "refinement",
        "generation",
    ]


def test_profile_crud_endpoints_use_yaml_store(tmp_path, monkeypatch):
    workflow_dir = tmp_path / "profiles" / "workflow"
    domain_dir = tmp_path / "profiles" / "domain_expert"
    workflow_dir.mkdir(parents=True)
    domain_dir.mkdir(parents=True)
    (workflow_dir / "existing_file.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "api_existing",
                "name": "API Existing",
                "description": "Original description.",
                "category": "workflow",
                "activates_on": {"intent_codes": ["T1_DRAFT"]},
                "knowledge": [],
                "capabilities": {"skills": []},
                "tools": [],
                "guardrails": [],
                "outputs": [],
                "co_activates_with": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_store, "WORKFLOW_DIR", workflow_dir)
    monkeypatch.setattr(profile_store, "DOMAIN_EXPERT_DIR", domain_dir)
    profile_store.load_profiles.cache_clear()
    client = _client()

    list_response = client.get("/v1/profiles")
    assert list_response.status_code == 200
    assert list_response.json()["workflow"][0]["id"] == "api_existing"

    create_response = client.post(
        "/v1/profiles",
        json={
            "id": "api_domain",
            "name": "API Domain",
            "description": "Domain description.",
            "category": "domain_expert",
            "sector": "Technologie, Media & Telecom",
            "topic_keywords": ["AI"],
            "expertise_areas": ["Analysis"],
        },
    )
    assert create_response.status_code == 201
    assert create_response.json()["id"] == "api_domain"

    duplicate_response = client.post(
        "/v1/profiles",
        json={
            "id": "api_domain",
            "name": "Duplicate",
            "description": "Duplicate description.",
            "category": "workflow",
        },
    )
    assert duplicate_response.status_code == 409

    update_response = client.put(
        "/v1/profiles/workflow/api_existing",
        json={
            "id": "api_existing",
            "name": "API Updated",
            "description": "Updated description.",
            "category": "workflow",
            "activates_on_intent_codes": ["T2_COMPLIANCE"],
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "API Updated"
    assert (workflow_dir / "existing_file.yaml").is_file()
    assert not (workflow_dir / "api_existing.yaml").exists()

    delete_response = client.delete("/v1/profiles/domain_expert/api_domain")
    assert delete_response.status_code == 200
    assert delete_response.json()["id"] == "api_domain"

    missing_response = client.delete("/v1/profiles/domain_expert/api_domain")
    assert missing_response.status_code == 404
    profile_store.load_profiles.cache_clear()


def test_profile_crud_endpoint_validation_errors():
    client = _client()

    invalid_id = client.post(
        "/v1/profiles",
        json={
            "id": "Bad ID",
            "name": "Bad",
            "description": "Invalid id.",
            "category": "workflow",
        },
    )
    invalid_category = client.delete("/v1/profiles/invalid_category/bad_id")
    invalid_path_id = client.delete("/v1/profiles/workflow/Bad-ID")

    assert invalid_id.status_code == 422
    assert invalid_category.status_code == 422
    assert invalid_path_id.status_code == 422
