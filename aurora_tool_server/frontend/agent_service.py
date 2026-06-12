"""Small service layer for the OpenAI Agents SDK Streamlit page."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict
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

You decide the path. Use AURORA tools whenever the user asks to draft, refine,
inspect, ground, research, or evaluate editorial content.

Hard policies:
- For normal drafting, use granular stage tools in this order:
  aurora_classify_intent -> aurora_select_profiles -> aurora_retrieve_context ->
  aurora_refine_prompt -> aurora_generate_draft -> aurora_evaluate_draft.
- Pass the exact full output objects from each stage into the next stage. Do not
  reconstruct, summarize, or shorten intent, profile, retrieval, snippet, or
  content JSON before passing it to another AURORA tool.
- Use aurora_run_pipeline_fallback only when the user explicitly asks for a
  quick full-pipeline run, or when a granular stage fails and fallback is the
  smallest recovery step.
- Never generate a draft without retrieved snippets.
- If the request needs clarification, first classify, select profiles, retrieve,
  then call aurora_refine_prompt with empty answers; relay the returned questions
  to the user.
- After the user answers clarification questions, call aurora_refine_prompt,
  with regenerate_on_pivot=true, then aurora_generate_draft, then
  aurora_evaluate_draft.
- For every generated draft, call aurora_evaluate_draft before presenting the
  answer as final.
- If evaluation returns passed=false or any failed_blocking items, regenerate exactly once
  using the same intent, profiles, and retrieved snippets, and append the
  evaluation feedback to the refined prompt. Then call aurora_evaluate_draft once more
  on the regenerated draft.
- After one regeneration attempt, stop. If the second evaluation still fails,
  present the regenerated draft as not approved, list the blocking failures,
  and explain what needs human/editorial repair.
- For review-only requests, evaluate only; do not generate a replacement draft
  unless the user asks.
- For research-only requests, retrieve only; do not generate a draft.
- Never present a draft as approved when evaluation passed=false or blocking
  failures exist.
- Always describe Tier-3/dCLP items as awaiting human signoff.

Use the smaller tools when the user asks to inspect intent, profiles, retrieval,
server health, or audit details. If a tool returns an error, explain the issue
and suggest the smallest recovery step. Do not invent citations, profile
details, KPI results, or audit events.

Mirror the user's language whenever possible. If the user writes in Dutch, or
an AURORA tool returns intent.language="nl", answer in Dutch and preserve Dutch
drafts without translating them back to English.
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
    max_turns: int = 12

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
        run_id: str | None = None,
    ) -> "AuroraAgentSettings":
        return cls(
            tool_config=AuroraAgentToolConfig(
                api_base_url=api_base_url,
                retrieval_backend=retrieval_backend,
                k=k,
                channel=channel,
                origin=origin,
                strict_mode=strict_mode,
                run_id=run_id,
            ),
            model=(model or os.getenv("AURORA_AGENT_MODEL") or DEFAULT_AGENT_MODEL).strip()
            or DEFAULT_AGENT_MODEL,
        )


@dataclass
class AuroraAgentTurnResult:
    final_output: str
    input_items: list[dict[str, Any]] = field(default_factory=list)
    last_agent_name: str = "AURORA Editorial Agent"
    tool_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AuroraAgentToolEvent:
    kind: str
    tool_name: str
    label: str
    summary: str
    status: str
    call_id: str | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)


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
    on_tool_event: Callable[[AuroraAgentToolEvent], None] | None = None,
) -> AuroraAgentTurnResult:
    _ensure_ready()
    assert Runner is not None

    agent = build_agent(settings)
    run_input: str | list[dict[str, Any]]
    if input_items:
        run_input = [*input_items, {"role": "user", "content": user_message}]
    else:
        run_input = user_message

    tool_events: list[AuroraAgentToolEvent] = []

    def record_event(event: AuroraAgentToolEvent) -> None:
        tool_events.append(event)
        if on_tool_event is not None:
            on_tool_event(event)

    try:
        result = asyncio.run(
            _run_streamed_agent_turn(
                agent,
                run_input,
                settings=settings,
                on_tool_event=record_event,
            )
        )
    except AgentsException as exc:
        raise AuroraAgentError(f"OpenAI agent run failed: {exc}") from exc
    except Exception as exc:
        raise AuroraAgentError(f"OpenAI agent run failed: {exc}") from exc

    return AuroraAgentTurnResult(
        final_output=str(result.final_output),
        input_items=_json_safe(result.to_input_list()),
        last_agent_name=getattr(result.last_agent, "name", "AURORA Editorial Agent"),
        tool_events=[asdict(event) for event in tool_events],
    )


async def _run_streamed_agent_turn(
    agent: Any,
    run_input: str | list[dict[str, Any]],
    *,
    settings: AuroraAgentSettings,
    on_tool_event: Callable[[AuroraAgentToolEvent], None],
) -> Any:
    assert Runner is not None

    result = Runner.run_streamed(agent, run_input, max_turns=settings.max_turns)
    call_names: dict[str, str] = {}
    async for event in result.stream_events():
        tool_event = _tool_event_from_stream_event(event, settings, call_names)
        if tool_event is not None:
            on_tool_event(tool_event)
    run_loop_exception = getattr(result, "run_loop_exception", None)
    if run_loop_exception:
        raise run_loop_exception
    return result


def _tool_event_from_stream_event(
    event: Any,
    settings: AuroraAgentSettings,
    call_names: dict[str, str],
) -> AuroraAgentToolEvent | None:
    if getattr(event, "type", None) != "run_item_stream_event":
        return None
    item = getattr(event, "item", None)
    item_type = getattr(item, "type", None)
    event_name = getattr(event, "name", "")
    call_id = _item_call_id(item)
    if item_type == "tool_call_item" or event_name == "tool_called":
        tool_name = _tool_name(item) or "tool"
        if call_id:
            call_names[call_id] = tool_name
        summary = _summarize_tool_call(tool_name, _item_arguments(item), settings)
        return AuroraAgentToolEvent(
            kind="call",
            tool_name=tool_name,
            label=f"-> {tool_name}",
            summary=summary,
            status="running",
            call_id=call_id,
        )
    if item_type == "tool_call_output_item" or event_name == "tool_output":
        tool_name = call_names.get(call_id or "", _tool_name(item) or "tool")
        output = _item_output(item)
        ok = _tool_output_ok(output)
        return AuroraAgentToolEvent(
            kind="output",
            tool_name=tool_name,
            label=f"<- {tool_name}",
            summary=_summarize_tool_output(tool_name, output),
            status="complete" if ok else "error",
            call_id=call_id,
            questions=_extract_tool_questions(tool_name, output),
        )
    return None


def _tool_name(item: Any) -> str | None:
    candidate = getattr(item, "tool_name", None)
    if candidate:
        return str(candidate)
    raw_item = getattr(item, "raw_item", None)
    if isinstance(raw_item, dict):
        value = raw_item.get("name") or raw_item.get("tool_name")
        return str(value) if value is not None else None
    value = getattr(raw_item, "name", None) or getattr(raw_item, "tool_name", None)
    return str(value) if value is not None else None


def _item_call_id(item: Any) -> str | None:
    value = getattr(item, "call_id", None)
    if value:
        return str(value)
    raw_item = getattr(item, "raw_item", None)
    if isinstance(raw_item, dict):
        value = raw_item.get("call_id") or raw_item.get("id")
        return str(value) if value is not None else None
    value = getattr(raw_item, "call_id", None) or getattr(raw_item, "id", None)
    return str(value) if value is not None else None


def _item_arguments(item: Any) -> Any:
    raw_item = getattr(item, "raw_item", None)
    if isinstance(raw_item, dict):
        return raw_item.get("arguments") or raw_item.get("params") or raw_item.get("input")
    return (
        getattr(raw_item, "arguments", None)
        or getattr(raw_item, "params", None)
        or getattr(raw_item, "input", None)
    )


def _item_output(item: Any) -> Any:
    return getattr(item, "output", None)


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _tool_output_ok(output: Any) -> bool:
    parsed = _parse_jsonish(output)
    if isinstance(parsed, dict) and "ok" in parsed:
        return bool(parsed.get("ok"))
    return True


def _summarize_tool_call(
    tool_name: str,
    arguments: Any,
    settings: AuroraAgentSettings,
) -> str:
    args = _parse_jsonish(arguments)
    if not isinstance(args, dict):
        return _short_text(str(args or ""))
    if tool_name == "aurora_retrieve_context":
        return (
            f"k={settings.tool_config.k}, "
            f"backend={settings.tool_config.retrieval_backend}, "
            f"prompt={_short_text(args.get('user_prompt', ''))}"
        )
    if tool_name == "aurora_run_pipeline_fallback":
        policy = args.get("refinement_policy", "skip")
        return (
            f"fallback full pipeline, refinement_policy={policy}, "
            f"prompt={_short_text(args.get('user_prompt', ''))}"
        )
    if tool_name == "aurora_select_profiles":
        intent = args.get("intent") or {}
        if isinstance(intent, dict):
            return f"tasks={intent.get('task_codes')}, sector={intent.get('sector')}"
        return "selecting profiles from intent"
    if tool_name == "aurora_refine_prompt":
        answers = args.get("answers") or {}
        return (
            f"answers={len(answers) if isinstance(answers, dict) else 0}, "
            f"regenerate_on_pivot={bool(args.get('regenerate_on_pivot'))}"
        )
    if tool_name == "aurora_generate_draft":
        snippets = args.get("snippets") or []
        return f"snippets={len(snippets) if isinstance(snippets, list) else 0}"
    if tool_name == "aurora_evaluate_draft":
        return (
            f"channel={settings.tool_config.channel}, "
            f"origin={settings.tool_config.origin}, "
            f"strict={settings.tool_config.strict_mode}"
        )
    if "user_prompt" in args:
        return f"prompt={_short_text(args.get('user_prompt', ''))}"
    return _short_text(", ".join(sorted(args.keys())) or "starting")


def _summarize_tool_output(tool_name: str, output: Any) -> str:
    parsed = _parse_jsonish(output)
    if isinstance(parsed, dict) and parsed.get("ok") is False:
        return f"failed: {_short_text(parsed.get('error', 'unknown error'), 180)}"
    result = parsed.get("result") if isinstance(parsed, dict) and "result" in parsed else parsed
    if not isinstance(result, dict):
        return _short_text(str(result or "done"), 180)
    if tool_name == "aurora_run_pipeline_fallback":
        refinement = result.get("refinement") or {}
        evaluation = result.get("evaluation") or {}
        return (
            f"fallback status={result.get('status')}, run_id={result.get('run_id')}, "
            f"questions={len(refinement.get('questions') or [])}, "
            f"passed={evaluation.get('passed') if evaluation else 'n/a'}"
        )
    if tool_name == "aurora_retrieve_context":
        retrieval = result.get("retrieval") or result
        return f"snippets={len(retrieval.get('snippets') or [])}, provider={retrieval.get('provider')}"
    if tool_name == "aurora_refine_prompt":
        return (
            f"done={result.get('done')}, questions={len(result.get('questions') or [])}, "
            f"needs_re_retrieval={result.get('needs_re_retrieval')}"
        )
    if tool_name == "aurora_generate_draft":
        return (
            f"body_chars={len(result.get('body') or '')}, "
            f"citations={len(result.get('citations') or [])}"
        )
    if tool_name == "aurora_evaluate_draft":
        return (
            f"passed={result.get('passed')}, "
            f"blocking={len(result.get('failed_blocking') or [])}, "
            f"tier3={len(result.get('dclp_steps_required') or [])}"
        )
    if tool_name == "aurora_classify_intent":
        return f"tasks={result.get('task_codes')}, language={result.get('language')}"
    if tool_name == "aurora_select_profiles":
        profiles = result.get("profiles") or result
        return (
            f"workflow={len(profiles.get('workflow') or [])}, "
            f"experts={len(profiles.get('domain_expert') or [])}"
        )
    return _short_text(", ".join(sorted(result.keys())) or "done", 180)


def _extract_tool_questions(tool_name: str, output: Any) -> list[dict[str, Any]]:
    parsed = _parse_jsonish(output)
    if isinstance(parsed, dict) and parsed.get("ok") is False:
        return []
    result = parsed.get("result") if isinstance(parsed, dict) and "result" in parsed else parsed
    if not isinstance(result, dict):
        return []

    if tool_name == "aurora_refine_prompt":
        return _normalize_questions(result.get("questions") or [])
    if tool_name == "aurora_run_pipeline_fallback":
        refinement = result.get("refinement") or {}
        if isinstance(refinement, dict):
            return _normalize_questions(refinement.get("questions") or [])
    return []


def extract_clarification_questions(
    content: str,
    tool_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    for event in tool_events or []:
        questions = _normalize_questions(event.get("questions") or [])
        if questions:
            return questions
    return parse_clarification_questions(content)


def parse_clarification_questions(content: str) -> list[dict[str, Any]]:
    """Parse common plain-text clarification lists into UI-friendly questions."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    option_re = re.compile(r"^(?:[-*]\s*)?([A-Za-z])[\).:]\s*(.+)$")
    numbered_re = re.compile(r"^(?:\d+[\).]\s*)?(.+\?)$")

    for line in lines:
        option_match = option_re.match(line)
        if option_match and current is not None:
            current.setdefault("choices", []).append(option_match.group(2).strip())
            continue

        question_text: str | None = None
        numbered_match = numbered_re.match(line)
        if numbered_match:
            question_text = numbered_match.group(1).strip()
        elif line.endswith("?") or _looks_like_short_question_heading(line):
            question_text = line.rstrip(":").strip()

        if question_text:
            if current is not None:
                questions.append(current)
            current = {"question": question_text, "choices": []}

    if current is not None:
        questions.append(current)

    return [
        question
        for question in _normalize_questions(questions)
        if question.get("choices")
    ][:5]


def _looks_like_short_question_heading(line: str) -> bool:
    if len(line) > 90 or line.lower().startswith(("please ", "reply ", "i have ")):
        return False
    lowered = line.rstrip(":").lower()
    return lowered in {
        "desired length",
        "tone",
        "target audience",
        "audience",
        "focus",
        "language",
        "format",
    } or lowered.startswith(("desired ", "what specific ", "which "))


def _normalize_questions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    questions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        choices = [
            str(choice).strip()
            for choice in (item.get("choices") or [])
            if str(choice).strip()
        ]
        questions.append({"question": question, "choices": choices[:6]})
    return questions[:5]


def _short_text(value: Any, limit: int = 96) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


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
