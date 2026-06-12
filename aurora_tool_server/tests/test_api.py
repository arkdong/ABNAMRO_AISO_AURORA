from fastapi.testclient import TestClient

from aurora_tool_server import AuroraConfig, AuroraCore
from aurora_tool_server import api


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

    audit = client.get(f"/v1/runs/{run_id}/audit")
    assert audit.status_code == 200
    assert [event["stage"] for event in audit.json()["events"]] == [
        "intent",
        "profiles",
        "retrieval",
        "refinement",
    ]
