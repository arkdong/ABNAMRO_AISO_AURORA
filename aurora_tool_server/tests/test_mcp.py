from aurora_tool_server import AuroraConfig, AuroraCore
from aurora_tool_server import mcp_server


def _use_deterministic_core() -> None:
    mcp_server.core = AuroraCore(
        AuroraConfig(
            intent_api_key=None,
            profile_api_key=None,
            retrieval_api_key=None,
            refinement_api_key=None,
            content_api_key=None,
            evaluation_api_key=None,
        )
    )


def test_mcp_classify_handler_returns_json_safe_result():
    _use_deterministic_core()
    payload = {
        "user_prompt": "Write an English article about agentic AI for TMT companies."
    }
    result = mcp_server.aurora_classify_intent_handler(payload)

    assert result["task_codes"][0] == "T1_DRAFT"
    assert result["source"] == "deterministic"


def test_mcp_run_pipeline_handler_returns_audit():
    _use_deterministic_core()
    run_id = "run_mcp_pipeline"
    payload = {
        "user_prompt": "Write an English article about cybersecurity for TMT companies.",
        "options": {"k": 2},
        "run_id": run_id,
    }
    result = mcp_server.aurora_run_pipeline_handler(payload)

    assert result["status"] == "completed"
    assert result["run_id"] == run_id
    assert result["audit"]["events"]
    assert {event["run_id"] for event in result["audit"]["events"]} == {run_id}
