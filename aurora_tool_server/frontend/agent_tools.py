"""OpenAI Agents SDK tools backed by the AURORA REST API."""

from __future__ import annotations

from copy import deepcopy
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
    run_id: str | None = None

    def stage_options(self) -> dict[str, Any]:
        return {
            "retrieval_backend": self.retrieval_backend,
            "k": int(self.k),
            "channel": self.channel,
            "origin": self.origin,
            "strict_mode": bool(self.strict_mode),
        }


_MAX_STAGE_CACHE_RUNS = 32
_STAGE_CACHE: dict[str, dict[str, Any]] = {}
_STAGE_CACHE_ORDER: list[str] = []


def clear_agent_tool_cache(run_id: str | None = None) -> None:
    """Clear cached full stage outputs used to repair agent handoffs."""
    if run_id is None:
        _STAGE_CACHE.clear()
        _STAGE_CACHE_ORDER.clear()
        return
    _STAGE_CACHE.pop(run_id, None)
    if run_id in _STAGE_CACHE_ORDER:
        _STAGE_CACHE_ORDER.remove(run_id)


def _effective_run_id(config: AuroraAgentToolConfig, run_id: str | None = None) -> str | None:
    return run_id if run_id is not None else config.run_id


def _cache_for(
    config: AuroraAgentToolConfig,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    key = _effective_run_id(config, run_id)
    if not key:
        return None
    if key not in _STAGE_CACHE:
        _STAGE_CACHE[key] = {}
        _STAGE_CACHE_ORDER.append(key)
        while len(_STAGE_CACHE_ORDER) > _MAX_STAGE_CACHE_RUNS:
            expired = _STAGE_CACHE_ORDER.pop(0)
            _STAGE_CACHE.pop(expired, None)
    return _STAGE_CACHE[key]


def _remember(
    config: AuroraAgentToolConfig,
    run_id: str | None = None,
    **stages: Any,
) -> None:
    cache = _cache_for(config, run_id)
    if cache is None:
        return
    for name, value in stages.items():
        if value is not None:
            cache[name] = deepcopy(value)


def _cached(config: AuroraAgentToolConfig, run_id: str | None, name: str) -> Any:
    cache = _cache_for(config, run_id)
    if cache is None or name not in cache:
        return None
    return deepcopy(cache[name])


def _unwrap_tool_result(value: Any) -> Any:
    if isinstance(value, dict) and value.get("ok") is True and "result" in value:
        return value["result"]
    return value


def _stage_payload(value: Any, stage_name: str) -> Any:
    payload = _unwrap_tool_result(value)
    if isinstance(payload, dict) and stage_name in payload:
        return payload[stage_name]
    return payload


def _is_full_intent(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("role"), str)
        and isinstance(value.get("task_codes"), list)
        and bool(value.get("task_codes"))
        and "confidence" in value
        and isinstance(value.get("task_reason"), str)
    )


def _is_full_profile_bundle(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for group_name in ("workflow", "domain_expert"):
        group = value.get(group_name, [])
        if not isinstance(group, list):
            return False
        for profile in group:
            if not isinstance(profile, dict):
                return False
            for key in ("id", "name", "description", "category"):
                if not profile.get(key):
                    return False
    return True


def _is_full_retrieval(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("query"), dict)
        and isinstance(value.get("snippets"), list)
        and isinstance(value.get("provider"), str)
    )


def _is_full_snippet(value: Any) -> bool:
    return isinstance(value, dict) and all(
        value.get(key) is not None
        for key in ("source_doc", "node_id", "title", "content", "score", "reason")
    )


def _is_full_snippets(value: Any) -> bool:
    return isinstance(value, list) and all(_is_full_snippet(item) for item in value)


def _is_full_content(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("body"), str)


def _coalesce_stage(
    config: AuroraAgentToolConfig,
    run_id: str | None,
    stage_name: str,
    value: Any,
    validator,
) -> Any:
    payload = _stage_payload(value, stage_name)
    if validator(payload):
        return payload
    cached = _cached(config, run_id, stage_name)
    if validator(cached):
        return cached
    return payload


def _valid_or_none(value: Any, validator) -> Any:
    return value if validator(value) else None


def _coalesce_snippets(
    config: AuroraAgentToolConfig,
    run_id: str | None,
    snippets: Any,
) -> list[dict[str, Any]]:
    payload = _stage_payload(snippets, "snippets")
    cached_retrieval = _cached(config, run_id, "retrieval")
    cached_snippets = (
        cached_retrieval.get("snippets")
        if isinstance(cached_retrieval, dict)
        else _cached(config, run_id, "snippets")
    )
    if _is_full_snippets(payload) and (payload or not cached_snippets):
        return payload
    if _is_full_snippets(cached_snippets):
        return cached_snippets
    return payload if isinstance(payload, list) else []


def aurora_health_check(config: AuroraAgentToolConfig) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        return client.health()
    finally:
        client.close()


def aurora_classify_intent(
    config: AuroraAgentToolConfig,
    user_prompt: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        intent = client.classify_intent(
            user_prompt,
            config.stage_options(),
            run_id=effective_run_id,
        )
        _remember(config, effective_run_id, intent=intent)
        return intent
    finally:
        client.close()


def aurora_select_profiles(
    config: AuroraAgentToolConfig,
    intent: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        intent = _coalesce_stage(
            config,
            effective_run_id,
            "intent",
            intent,
            _is_full_intent,
        )
        profiles = client.select_profiles(
            intent,
            config.stage_options(),
            run_id=effective_run_id,
        )
        _remember(config, effective_run_id, intent=intent, profiles=profiles)
        return profiles
    finally:
        client.close()


def aurora_retrieve_context(
    config: AuroraAgentToolConfig,
    user_prompt: str,
    intent: dict[str, Any],
    profiles: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        options = config.stage_options()
        intent = _coalesce_stage(
            config,
            effective_run_id,
            "intent",
            intent,
            _is_full_intent,
        )
        profiles = _coalesce_stage(
            config,
            effective_run_id,
            "profiles",
            profiles,
            _is_full_profile_bundle,
        )
        retrieval = client.retrieve_context(
            user_prompt,
            intent,
            profiles,
            options,
            run_id=effective_run_id,
        )
        _remember(
            config,
            effective_run_id,
            intent=intent,
            profiles=profiles,
            retrieval=retrieval,
            user_prompt=user_prompt,
        )
        return retrieval
    finally:
        client.close()


def aurora_run_pipeline(
    config: AuroraAgentToolConfig,
    user_prompt: str,
    refinement_policy: str = "skip",
    run_id: str | None = None,
) -> dict[str, Any]:
    if refinement_policy not in {"skip", "ask_first"}:
        raise ValueError("refinement_policy must be 'skip' or 'ask_first'")

    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        result = client.run_pipeline(
            user_prompt,
            refinement_policy=refinement_policy,
            options=config.stage_options(),
            run_id=effective_run_id,
        )
        _remember(
            config,
            effective_run_id,
            intent=result.get("intent"),
            profiles=result.get("profiles"),
            retrieval=result.get("retrieval"),
            refinement=result.get("refinement"),
            content=result.get("content"),
            evaluation=result.get("evaluation"),
            user_prompt=user_prompt,
        )
        return result
    finally:
        client.close()


def aurora_refine_prompt(
    config: AuroraAgentToolConfig,
    user_prompt: str,
    intent: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    retrieval: dict[str, Any] | None = None,
    answers: dict[str, str] | None = None,
    regenerate_on_pivot: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        intent = _coalesce_stage(
            config,
            effective_run_id,
            "intent",
            intent,
            _is_full_intent,
        )
        profiles = _coalesce_stage(
            config,
            effective_run_id,
            "profiles",
            profiles,
            _is_full_profile_bundle,
        )
        retrieval = _coalesce_stage(
            config,
            effective_run_id,
            "retrieval",
            retrieval,
            _is_full_retrieval,
        )
        intent = _valid_or_none(intent, _is_full_intent)
        profiles = _valid_or_none(profiles, _is_full_profile_bundle)
        retrieval = _valid_or_none(retrieval, _is_full_retrieval)
        refinement = client.refine_prompt(
            user_prompt,
            intent=intent,
            profiles=profiles,
            retrieval=retrieval,
            answers=answers or {},
            regenerate_on_pivot=regenerate_on_pivot,
            options=config.stage_options(),
            run_id=effective_run_id,
        )
        _remember(
            config,
            effective_run_id,
            intent=refinement.get("new_intent") or intent,
            profiles=refinement.get("profiles") or profiles,
            retrieval=refinement.get("retrieval") or retrieval,
            refinement=refinement,
            refined_prompt=refinement.get("refined_prompt"),
            user_prompt=user_prompt,
        )
        return refinement
    finally:
        client.close()


def aurora_generate_draft(
    config: AuroraAgentToolConfig,
    refined_prompt: str,
    intent: dict[str, Any],
    profiles: dict[str, Any],
    snippets: list[dict[str, Any]],
    run_id: str | None = None,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        intent = _coalesce_stage(
            config,
            effective_run_id,
            "intent",
            intent,
            _is_full_intent,
        )
        profiles = _coalesce_stage(
            config,
            effective_run_id,
            "profiles",
            profiles,
            _is_full_profile_bundle,
        )
        snippets = _coalesce_snippets(config, effective_run_id, snippets)
        content = client.generate_draft(
            refined_prompt=refined_prompt,
            intent=intent,
            profiles=profiles,
            snippets=snippets,
            options=config.stage_options(),
            run_id=effective_run_id,
        )
        _remember(
            config,
            effective_run_id,
            intent=intent,
            profiles=profiles,
            snippets=snippets,
            refined_prompt=refined_prompt,
            content=content,
        )
        return content
    finally:
        client.close()


def aurora_evaluate_draft(
    config: AuroraAgentToolConfig,
    refined_prompt: str,
    content: dict[str, Any],
    intent: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    snippets: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    client = AuroraApiClient(config.api_base_url)
    try:
        effective_run_id = _effective_run_id(config, run_id)
        intent = _coalesce_stage(
            config,
            effective_run_id,
            "intent",
            intent,
            _is_full_intent,
        )
        profiles = _coalesce_stage(
            config,
            effective_run_id,
            "profiles",
            profiles,
            _is_full_profile_bundle,
        )
        content = _coalesce_stage(
            config,
            effective_run_id,
            "content",
            content,
            _is_full_content,
        )
        snippets = _coalesce_snippets(config, effective_run_id, snippets or [])
        evaluation = client.evaluate_draft(
            refined_prompt=refined_prompt,
            content=content,
            intent=intent,
            profiles=profiles,
            snippets=snippets or [],
            options=config.stage_options(),
            run_id=effective_run_id,
        )
        _remember(
            config,
            effective_run_id,
            intent=intent,
            profiles=profiles,
            snippets=snippets,
            content=content,
            refined_prompt=refined_prompt,
            evaluation=evaluation,
        )
        return evaluation
    finally:
        client.close()


def aurora_get_audit_trace(
    config: AuroraAgentToolConfig,
    run_id: str | None = None,
) -> dict[str, Any]:
    effective_run_id = _effective_run_id(config, run_id)
    if not effective_run_id:
        raise ValueError("run_id is required when the agent session has no run_id")
    client = AuroraApiClient(config.api_base_url)
    try:
        return client.get_audit_trace(effective_run_id)
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
    def classify_tool(user_prompt: str, run_id: str | None = None) -> dict[str, Any]:
        """Classify a user request into AURORA editorial intent metadata."""

        return _tool_result(lambda: aurora_classify_intent(config, user_prompt, run_id))

    @function_tool(name_override="aurora_select_profiles", strict_mode=False)
    def profiles_tool(intent: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
        """Select matching workflow/domain profiles for a classified intent."""

        return _tool_result(lambda: aurora_select_profiles(config, intent, run_id))

    @function_tool(name_override="aurora_retrieve_context", strict_mode=False)
    def retrieval_tool(
        user_prompt: str,
        intent: dict[str, Any],
        profiles: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve grounding snippets for a prompt using classified intent and profiles."""

        return _tool_result(
            lambda: aurora_retrieve_context(config, user_prompt, intent, profiles, run_id)
        )

    @function_tool(name_override="aurora_run_pipeline_fallback", strict_mode=False)
    def pipeline_tool(
        user_prompt: str,
        refinement_policy: str = "skip",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Fallback-only full AURORA pipeline when granular stage tools cannot recover."""

        return _tool_result(
            lambda: aurora_run_pipeline(config, user_prompt, refinement_policy, run_id)
        )

    @function_tool(name_override="aurora_refine_prompt", strict_mode=False)
    def refine_tool(
        user_prompt: str,
        intent: dict[str, Any] | None = None,
        profiles: dict[str, Any] | None = None,
        retrieval: dict[str, Any] | None = None,
        answers: dict[str, str] | None = None,
        regenerate_on_pivot: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Refine a prompt after clarification answers; regenerate retrieval on pivots."""

        return _tool_result(
            lambda: aurora_refine_prompt(
                config,
                user_prompt,
                intent,
                profiles,
                retrieval,
                answers,
                regenerate_on_pivot,
                run_id,
            )
        )

    @function_tool(name_override="aurora_generate_draft", strict_mode=False)
    def generate_tool(
        refined_prompt: str,
        intent: dict[str, Any],
        profiles: dict[str, Any],
        snippets: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a grounded draft from a refined prompt and retrieved snippets."""

        return _tool_result(
            lambda: aurora_generate_draft(
                config,
                refined_prompt,
                intent,
                profiles,
                snippets,
                run_id,
            )
        )

    @function_tool(name_override="aurora_evaluate_draft", strict_mode=False)
    def evaluate_tool(
        refined_prompt: str,
        content: dict[str, Any],
        intent: dict[str, Any] | None = None,
        profiles: dict[str, Any] | None = None,
        snippets: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a generated or user-supplied draft against AURORA KPI checks."""

        return _tool_result(
            lambda: aurora_evaluate_draft(
                config,
                refined_prompt,
                content,
                intent,
                profiles,
                snippets,
                run_id,
            )
        )

    @function_tool(name_override="aurora_get_audit_trace", strict_mode=False)
    def audit_tool(run_id: str | None = None) -> dict[str, Any]:
        """Fetch the audit trace for a run; defaults to the current agent session."""

        return _tool_result(lambda: aurora_get_audit_trace(config, run_id))

    return [
        health_tool,
        classify_tool,
        profiles_tool,
        retrieval_tool,
        pipeline_tool,
        refine_tool,
        generate_tool,
        evaluate_tool,
        audit_tool,
    ]


def _tool_result(call) -> dict[str, Any]:
    try:
        return {"ok": True, "result": call()}
    except Exception as exc:  # The model can recover when tool errors are data.
        return {"ok": False, "error": str(exc)}
