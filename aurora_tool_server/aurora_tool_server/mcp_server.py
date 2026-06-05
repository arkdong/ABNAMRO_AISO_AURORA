"""MCP tool surface for the standalone AURORA server."""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - local fallback when dependency is absent
    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, Any] = {}

        def tool(self, func=None, **_kwargs):
            def decorator(inner):
                self.tools[inner.__name__] = inner
                return inner

            if func is not None:
                return decorator(func)
            return decorator

        def run(self) -> None:
            raise RuntimeError("The official mcp package is not installed.")

from .core import AuroraConfig, AuroraCore
from .schemas import (
    EvaluateRequest,
    GenerateRequest,
    IntentRequest,
    ProfileRequest,
    RefineRequest,
    RetrievalRequest,
    RunRequest,
)

mcp = FastMCP("aurora-tool-server")
core = AuroraCore(AuroraConfig.from_env())


def _dump(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


def aurora_classify_intent_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = IntentRequest(**payload)
    return _dump(core.classify_intent(request.user_prompt, options=request.options))


def aurora_select_profiles_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = ProfileRequest(**payload)
    return _dump(core.select_profiles(request.intent, options=request.options))


def aurora_retrieve_context_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = RetrievalRequest(**payload)
    return _dump(
        core.retrieve_context(
            request.user_prompt,
            request.intent,
            request.profiles,
            options=request.options,
        )
    )


def aurora_refine_prompt_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = RefineRequest(**payload)
    intent = request.intent or core.classify_intent(request.user_prompt, options=request.options)
    profiles = request.profiles or core.select_profiles(intent, options=request.options)
    retrieval = request.retrieval or core.retrieve_context(
        request.user_prompt,
        intent,
        profiles,
        options=request.options,
    )
    return _dump(
        core.refine_prompt(
            request.user_prompt,
            intent,
            profiles,
            retrieval,
            answers=request.answers,
            regenerate_on_pivot=request.regenerate_on_pivot,
            ask_questions=not bool(request.answers),
            options=request.options,
        )
    )


def aurora_generate_draft_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = GenerateRequest(**payload)
    return _dump(
        core.generate_draft(
            request.refined_prompt,
            request.intent,
            request.profiles,
            request.snippets,
            options=request.options,
        )
    )


def aurora_evaluate_draft_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = EvaluateRequest(**payload)
    return _dump(
        core.evaluate_draft(
            request.refined_prompt,
            request.content,
            request.snippets,
            options=request.options,
        )
    )


def aurora_run_pipeline_handler(payload: dict[str, Any]) -> dict[str, Any]:
    request = RunRequest(**payload)
    return _dump(
        core.run_pipeline(
            request.user_prompt,
            refinement_policy=request.refinement_policy,
            options=request.options,
        )
    )


def aurora_get_audit_trace_handler(run_id: str) -> dict[str, Any]:
    return _dump(core.get_audit_trace(run_id))


@mcp.tool()
def aurora_classify_intent(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_classify_intent_handler(payload)


@mcp.tool()
def aurora_select_profiles(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_select_profiles_handler(payload)


@mcp.tool()
def aurora_retrieve_context(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_retrieve_context_handler(payload)


@mcp.tool()
def aurora_refine_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_refine_prompt_handler(payload)


@mcp.tool()
def aurora_generate_draft(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_generate_draft_handler(payload)


@mcp.tool()
def aurora_evaluate_draft(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_evaluate_draft_handler(payload)


@mcp.tool()
def aurora_run_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    return aurora_run_pipeline_handler(payload)


@mcp.tool()
def aurora_get_audit_trace(run_id: str) -> dict[str, Any]:
    return aurora_get_audit_trace_handler(run_id)


if __name__ == "__main__":
    mcp.run()
