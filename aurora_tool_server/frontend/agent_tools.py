"""OpenAI Agents SDK tools backed by the AURORA REST API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # Streamlit page execution imports modules from frontend/ directly.
    from .api_client import AuroraApiClient
except ImportError:  # pragma: no cover - exercised by Streamlit runtime import mode.
    from api_client import AuroraApiClient


@dataclass(frozen=True)
class AuroraAgentToolConfig:
    api_base_url: str = "http://127.0.0.1:8000"
    retrieval_backend: str = "pageindex"
    k: int = 5
    channel: str = "web"
    origin: str = "instant"
    strict_mode: bool = False

    def stage_options(self) -> dict[str, Any]:
        return {
            "retrieval_backend": self.retrieval_backend,
            "k": int(self.k),
            "channel": self.channel,
            "origin": self.origin,
            "strict_mode": bool(self.strict_mode),
        }


def aurora_health_check(config: AuroraAgentToolConfig) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        return client.health()
    finally:
        client.close()


def aurora_classify_intent(
    config: AuroraAgentToolConfig,
    user_prompt: str,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        return client.classify_intent(user_prompt, config.stage_options())
    finally:
        client.close()


def aurora_select_profiles(
    config: AuroraAgentToolConfig,
    user_prompt: str,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        intent = client.classify_intent(user_prompt, config.stage_options())
        profiles = client.select_profiles(intent, config.stage_options())
        return {"intent": intent, "profiles": profiles}
    finally:
        client.close()


def aurora_retrieve_context(
    config: AuroraAgentToolConfig,
    user_prompt: str,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        options = config.stage_options()
        intent = client.classify_intent(user_prompt, options)
        profiles = client.select_profiles(intent, options)
        retrieval = client.retrieve_context(user_prompt, intent, profiles, options)
        return {"intent": intent, "profiles": profiles, "retrieval": retrieval}
    finally:
        client.close()


def aurora_run_pipeline(
    config: AuroraAgentToolConfig,
    user_prompt: str,
    refinement_policy: str = "skip",
) -> dict[str, Any]:
    if refinement_policy not in {"skip", "ask_first"}:
        raise ValueError("refinement_policy must be 'skip' or 'ask_first'")

    client = AuroraApiClient(config.api_base_url)
    try:
        return client.run_pipeline(
            user_prompt,
            refinement_policy=refinement_policy,
            options=config.stage_options(),
        )
    finally:
        client.close()


def aurora_get_audit_trace(
    config: AuroraAgentToolConfig,
    run_id: str,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        return client.get_audit_trace(run_id)
    finally:
        client.close()


def build_aurora_function_tools(config: AuroraAgentToolConfig) -> list[Any]:
    """Create SDK function tools with the runtime config closed over."""

    from agents import function_tool

    @function_tool(name_override="aurora_health_check", strict_mode=False)
    def health_tool() -> dict[str, Any]:
        """Check whether the AURORA tool server is reachable."""

        return _tool_result(lambda: aurora_health_check(config))

    @function_tool(name_override="aurora_classify_intent", strict_mode=False)
    def classify_tool(user_prompt: str) -> dict[str, Any]:
        """Classify a user request into AURORA editorial intent metadata."""

        return _tool_result(lambda: aurora_classify_intent(config, user_prompt))

    @function_tool(name_override="aurora_select_profiles", strict_mode=False)
    def profiles_tool(user_prompt: str) -> dict[str, Any]:
        """Classify a user request and select matching workflow/domain profiles."""

        return _tool_result(lambda: aurora_select_profiles(config, user_prompt))

    @function_tool(name_override="aurora_retrieve_context", strict_mode=False)
    def retrieval_tool(user_prompt: str) -> dict[str, Any]:
        """Classify, select profiles, and retrieve grounding snippets for a prompt."""

        return _tool_result(lambda: aurora_retrieve_context(config, user_prompt))

    @function_tool(name_override="aurora_run_pipeline", strict_mode=False)
    def pipeline_tool(
        user_prompt: str,
        refinement_policy: str = "skip",
    ) -> dict[str, Any]:
        """Run the full AURORA pipeline for a content request."""

        return _tool_result(lambda: aurora_run_pipeline(config, user_prompt, refinement_policy))

    @function_tool(name_override="aurora_get_audit_trace", strict_mode=False)
    def audit_tool(run_id: str) -> dict[str, Any]:
        """Fetch the in-memory audit trace for a previous AURORA run."""

        return _tool_result(lambda: aurora_get_audit_trace(config, run_id))

    return [
        health_tool,
        classify_tool,
        profiles_tool,
        retrieval_tool,
        pipeline_tool,
        audit_tool,
    ]


def _tool_result(call) -> dict[str, Any]:
    try:
        return {"ok": True, "result": call()}
    except Exception as exc:  # The model can recover when tool errors are data.
        return {"ok": False, "error": str(exc)}
