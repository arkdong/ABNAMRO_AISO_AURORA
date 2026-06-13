from __future__ import annotations

import streamlit as st

from agent_service import (
    DEFAULT_AGENT_MODEL,
    AuroraAgentError,
    AuroraAgentSettings,
    extract_clarification_questions,
    readiness_error,
    run_agent_turn,
)
from branding import apply_branding
from settings_state import init_agent_state


st.set_page_config(page_title="AURORA · ABN AMRO", layout="centered")
apply_branding("AURORA", "AI Agent Interface")


def _settings() -> AuroraAgentSettings:
    return AuroraAgentSettings.from_values(
        api_base_url=st.session_state["agent_api_base_url"],
        model=st.session_state["agent_model"],
        retrieval_backend=st.session_state["agent_retrieval_backend"],
        k=int(st.session_state["agent_retrieval_k"]),
        channel=st.session_state["agent_channel"],
        origin=st.session_state["agent_origin"],
        strict_mode=bool(st.session_state["agent_strict_mode"]),
        run_id=st.session_state["agent_run_id"],
    )


init_agent_state(DEFAULT_AGENT_MODEL)

st.caption(
    f"Agent-first chat · `{st.session_state['agent_api_base_url']}` · "
    f"audit `{st.session_state['agent_run_id']}`"
)

setup_error = readiness_error()
if setup_error:
    st.warning(setup_error)


def _render_tool_events(events: list[dict]) -> None:
    if not events:
        return
    with st.expander(f"Agent tool calls ({sum(1 for event in events if event.get('kind') == 'call')})"):
        for event in events:
            state = event.get("status", "complete")
            if state == "running":
                state = "complete"
            with st.status(event.get("label", event.get("tool_name", "tool")), state=state, expanded=False):
                summary = event.get("summary")
                if summary:
                    st.write(summary)


def _make_live_tool_renderer():
    status_blocks = {}
    fallback_index = 0

    def render(event) -> None:
        nonlocal fallback_index
        key = event.call_id or f"{event.tool_name}-{fallback_index}"
        if event.call_id is None:
            fallback_index += 1
        if event.kind == "call":
            status = st.status(event.label, state="running", expanded=True)
            status.write(event.summary)
            status_blocks[key] = status
            return
        status = status_blocks.get(key)
        if status is None:
            status = st.status(event.label, state="running", expanded=True)
            status_blocks[key] = status
        status.write(event.summary)
        status.update(label=event.label, state=event.status, expanded=False)

    return render


def _answers_prompt(questions: list[dict], answers: list[str], extra_context: str) -> str:
    lines = ["Clarification answers:"]
    for index, (question, answer) in enumerate(zip(questions, answers, strict=False), start=1):
        lines.append(f"{index}. {question.get('question', f'Question {index}')} {answer}")
    if extra_context.strip():
        lines.append("")
        lines.append(f"Additional context: {extra_context.strip()}")
    return "\n".join(lines)


def _run_agent_prompt(prompt: str) -> None:
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        live_tool_events = []
        render_live_tool_event = _make_live_tool_renderer()

        def on_tool_event(event) -> None:
            live_tool_events.append(dict(event.__dict__))
            render_live_tool_event(event)

        with st.spinner("Running AURORA agent..."):
            try:
                result = run_agent_turn(
                    prompt,
                    settings=_settings(),
                    input_items=st.session_state.agent_input_items,
                    on_tool_event=on_tool_event,
                )
            except AuroraAgentError as exc:
                content = f"Agent error: {exc}"
                st.error(content)
                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_events": live_tool_events,
                    }
                )
                return

        questions = extract_clarification_questions(result.final_output, result.tool_events)
        st.markdown(result.final_output)
        caption = f"Answered by {result.last_agent_name}"
        st.caption(caption)
        st.session_state.agent_input_items = result.input_items
        st.session_state.agent_messages.append(
            {
                "role": "assistant",
                "content": result.final_output,
                "caption": caption,
                "tool_events": result.tool_events,
                "clarification_questions": questions,
            }
        )


def _render_clarification_form(message_idx: int, message: dict) -> None:
    questions = message.get("clarification_questions") or extract_clarification_questions(
        message.get("content", ""),
        message.get("tool_events", []),
    )
    if not questions or message.get("clarification_answered"):
        return

    with st.chat_message("assistant"):
        with st.form(f"agent_clarification_{message_idx}"):
            answers: list[str] = []
            for index, question in enumerate(questions):
                label = question.get("question", f"Question {index + 1}")
                choices = question.get("choices") or []
                key_base = f"agent_clarification_{message_idx}_{index}"
                if choices:
                    selected = st.radio(label, choices, key=f"{key_base}_choice")
                    custom = st.text_input(
                        f"Custom answer {index + 1}",
                        key=f"{key_base}_custom",
                        label_visibility="collapsed",
                        placeholder="Custom answer",
                    )
                    answers.append(custom.strip() or selected)
                else:
                    answers.append(st.text_area(label, key=f"{key_base}_text", height=88).strip())

            extra_context = st.text_area(
                "Additional context",
                key=f"agent_clarification_{message_idx}_extra",
                height=88,
                placeholder="Optional extra context",
            )
            submitted = st.form_submit_button("Continue", type="primary")

    if not submitted:
        return
    if any(not answer.strip() for answer in answers):
        st.warning("Please answer every question before continuing.")
        return

    message["clarification_answered"] = True
    _run_agent_prompt(_answers_prompt(questions, answers, extra_context))
    st.rerun()


for message_idx, message in enumerate(st.session_state.agent_messages):
    with st.chat_message(message["role"]):
        _render_tool_events(message.get("tool_events", []))
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])
    if message.get("role") == "assistant":
        _render_clarification_form(message_idx, message)

prompt = st.chat_input("Ask the AURORA agent", disabled=setup_error is not None)

if prompt:
    _run_agent_prompt(prompt)
