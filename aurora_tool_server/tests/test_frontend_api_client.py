from __future__ import annotations

import httpx
import pytest

from frontend.api_client import AuroraApiClient, AuroraApiError


def _client(handler) -> AuroraApiClient:
    return AuroraApiClient(
        "http://aurora.test",
        transport=httpx.MockTransport(handler),
    )


def test_health_check():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    assert client.health() == {"status": "ok"}
    client.close()


def test_classify_select_retrieve_payloads_round_trip():
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        body = {} if not payload else __import__("json").loads(payload)
        seen.append((request.url.path, body))
        if request.url.path == "/v1/intent/classify":
            return httpx.Response(200, json={"role": "Insights Editorial", "task_codes": ["T1_DRAFT"], "confidence": 0.9, "task_reason": "test", "source": "deterministic"})
        if request.url.path == "/v1/profiles/select":
            return httpx.Response(200, json={"workflow": [], "domain_expert": []})
        if request.url.path == "/v1/retrieval/search":
            return httpx.Response(200, json={"query": body, "snippets": [], "provider": "pageindex", "corpora_searched": [], "source": "deterministic"})
        raise AssertionError(request.url.path)

    client = _client(handler)
    options = {"k": 3, "retrieval_backend": "pageindex"}
    intent = client.classify_intent("Write an article", options)
    profiles = client.select_profiles(intent)
    client.retrieve_context("Write an article", intent, profiles, options)

    assert seen[0] == (
        "/v1/intent/classify",
        {"user_prompt": "Write an article", "options": options},
    )
    assert seen[1] == ("/v1/profiles/select", {"intent": intent, "options": {}})
    assert seen[2][0] == "/v1/retrieval/search"
    assert seen[2][1]["profiles"] == profiles
    client.close()


def test_generation_and_evaluation_payloads():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        body = __import__("json").loads(request.read())
        if request.url.path == "/v1/drafts/generate":
            assert body["refined_prompt"] == "Draft it"
            assert body["snippets"][0]["node_id"] == "n1"
            return httpx.Response(200, json={"body": "Draft body [1]", "citations": [], "source": "deterministic"})
        if request.url.path == "/v1/evaluations/score":
            assert body["content"]["body"] == "Draft body [1]"
            return httpx.Response(200, json={"passed": True, "results": [], "channel": "web", "origin": "instant", "source": "deterministic"})
        raise AssertionError(request.url.path)

    client = _client(handler)
    intent = {"role": "Insights Editorial", "task_codes": ["T1_DRAFT"], "confidence": 0.9, "task_reason": "test", "source": "deterministic"}
    profiles = {"workflow": [], "domain_expert": []}
    snippets = [{"source_doc": "corpus", "node_id": "n1", "title": "T", "content": "C", "score": 1.0, "reason": "test"}]
    content = client.generate_draft(
        refined_prompt="Draft it",
        intent=intent,
        profiles=profiles,
        snippets=snippets,
        options={"channel": "web"},
    )
    evaluation = client.evaluate_draft(
        refined_prompt="Draft it",
        content=content,
        intent=intent,
        profiles=profiles,
        snippets=snippets,
        options={"channel": "web"},
    )

    assert paths == ["/v1/drafts/generate", "/v1/evaluations/score"]
    assert evaluation["passed"] is True
    client.close()


def test_run_pipeline_and_audit_payloads():
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        body = {} if not payload else __import__("json").loads(payload)
        seen.append((request.url.path, body))
        if request.url.path == "/v1/runs":
            return httpx.Response(
                200,
                json={
                    "run_id": "run_123",
                    "status": "completed",
                    "intent": {
                        "role": "Insights Editorial",
                        "task_codes": ["T1_DRAFT"],
                        "confidence": 0.9,
                        "task_reason": "test",
                    },
                    "profiles": {"workflow": [], "domain_expert": []},
                    "retrieval": {
                        "query": {"user_prompt": "Draft it", "k": 3, "retrieval_backend": "pageindex"},
                        "snippets": [],
                        "provider": "pageindex",
                        "corpora_searched": [],
                    },
                    "refinement": {
                        "original_prompt": "Draft it",
                        "refined_prompt": "Draft it",
                        "done": True,
                    },
                    "content": {"body": "Draft body", "citations": []},
                    "evaluation": {"passed": True, "results": []},
                    "audit": {"run_id": "run_123", "events": []},
                },
            )
        if request.url.path == "/v1/runs/run_123/audit":
            return httpx.Response(200, json={"run_id": "run_123", "events": []})
        raise AssertionError(request.url.path)

    client = _client(handler)
    options = {"k": 3, "retrieval_backend": "pageindex"}
    run = client.run_pipeline("Draft it", refinement_policy="skip", options=options)
    audit = client.get_audit_trace("run_123")

    assert run["run_id"] == "run_123"
    assert audit == {"run_id": "run_123", "events": []}
    assert seen[0] == (
        "/v1/runs",
        {"user_prompt": "Draft it", "refinement_policy": "skip", "options": options},
    )
    assert seen[1] == ("/v1/runs/run_123/audit", {})
    client.close()


def test_http_error_formatting_uses_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad payload"})

    client = _client(handler)
    with pytest.raises(AuroraApiError, match="422.*bad payload"):
        client.health()
    client.close()
