"""Small service layer for the OpenAI Agents SDK Streamlit page."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from aurora_tool_server.env import load_project_env
except ImportError:  # pragma: no cover - defensive fallback for direct script execution.
    from dotenv import load_dotenv

    def load_project_env() -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

load_project_env()

try:
    from agents import Agent, Runner
    from agents.exceptions import AgentsException
except Exception as exc:  # pragma: no cover - depends on optional runtime package.
    Agent = None  # type: ignore[assignment]
    Runner = None  # type: ignore[assignment]
    AgentsException = Exception  # type: ignore[assignment]
    AGENTS_IMPORT_ERROR: Exception | None = exc
else:
    AGENTS_IMPORT_ERROR = None

try:  # Streamlit page execution imports modules from frontend/ directly.
    from .agent_tools import AuroraAgentToolConfig, build_aurora_function_tools
except ImportError:  # pragma: no cover - exercised by Streamlit runtime import mode.
    from agent_tools import AuroraAgentToolConfig, build_aurora_function_tools


DEFAULT_AGENT_MODEL = "gpt-5-mini"

AGENT_INSTRUCTIONS = """
You are AURORA Editorial Agent, a calm editorial copilot for AURORA.

Use the AURORA tool server whenever the user asks to draft, refine, inspect,
ground, or evaluate editorial content. Prefer aurora_run_pipeline for full
content-generation requests. Use the smaller tools when the user asks to inspect
intent, profiles, retrieval, server health, or audit details.

When a tool returns content, summarize the result clearly and include practical
next steps. Preserve run_id values so the user can ask for an audit trace later.
If a tool returns an error, explain the issue and suggest the smallest recovery
step. Do not invent citations, profile details, KPI results, or audit events.
""".strip()


class AuroraAgentError(RuntimeError):
    """Base error for agent setup and runtime failures."""


class AuroraAgentDependencyError(AuroraAgentError):
    """Raised when the Agents SDK package cannot be imported."""


class AuroraAgentConfigurationError(AuroraAgentError):
    """Raised when required environment configuration is missing."""


@dataclass(frozen=True)
class AuroraAgentSettings:
    tool_config: AuroraAgentToolConfig
    model: str = DEFAULT_AGENT_MODEL
    max_turns: int = 8

    @classmethod
    def from_values(
        cls,
        *,
        api_base_url: str,
        model: str | None = None,
        retrieval_backend: str,
        k: int,
        channel: str,
        origin: str,
        strict_mode: bool,
    ) -> "AuroraAgentSettings":
        return cls(
            tool_config=AuroraAgentToolConfig(
                api_base_url=api_base_url,
                retrieval_backend=retrieval_backend,
                k=k,
                channel=channel,
                origin=origin,
                strict_mode=strict_mode,
            ),
            model=(model or os.getenv("AURORA_AGENT_MODEL") or DEFAULT_AGENT_MODEL).strip()
            or DEFAULT_AGENT_MODEL,
        )


@dataclass
class AuroraAgentTurnResult:
    final_output: str
    input_items: list[dict[str, Any]] = field(default_factory=list)
    last_agent_name: str = "AURORA Editorial Agent"


def is_agent_ready() -> bool:
    return AGENTS_IMPORT_ERROR is None and bool(os.getenv("OPENAI_API_KEY"))


def readiness_error() -> str | None:
    if AGENTS_IMPORT_ERROR is not None:
        return f"OpenAI Agents SDK is not available: {AGENTS_IMPORT_ERROR}"
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set for the Streamlit process."
    return None


def build_agent(settings: AuroraAgentSettings) -> Any:
    _ensure_ready()
    assert Agent is not None
    return Agent(
        name="AURORA Editorial Agent",
        instructions=AGENT_INSTRUCTIONS,
        model=settings.model,
        tools=build_aurora_function_tools(settings.tool_config),
    )


def run_agent_turn(
    user_message: str,
    *,
    settings: AuroraAgentSettings,
    input_items: list[dict[str, Any]] | None = None,
) -> AuroraAgentTurnResult:
    _ensure_ready()
    assert Runner is not None

    agent = build_agent(settings)
    run_input: str | list[dict[str, Any]]
    if input_items:
        run_input = [*input_items, {"role": "user", "content": user_message}]
    else:
        run_input = user_message

    try:
        result = Runner.run_sync(agent, run_input, max_turns=settings.max_turns)
    except AgentsException as exc:
        raise AuroraAgentError(f"OpenAI agent run failed: {exc}") from exc
    except Exception as exc:
        raise AuroraAgentError(f"OpenAI agent run failed: {exc}") from exc

    return AuroraAgentTurnResult(
        final_output=str(result.final_output),
        input_items=_json_safe(result.to_input_list()),
        last_agent_name=getattr(result.last_agent, "name", "AURORA Editorial Agent"),
    )


def _ensure_ready() -> None:
    if AGENTS_IMPORT_ERROR is not None:
        raise AuroraAgentDependencyError(
            f"OpenAI Agents SDK is not available: {AGENTS_IMPORT_ERROR}"
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise AuroraAgentConfigurationError(
            "OPENAI_API_KEY is not set for the Streamlit process."
        )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
