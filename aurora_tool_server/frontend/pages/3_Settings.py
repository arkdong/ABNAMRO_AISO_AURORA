"""Settings page for the AURORA Streamlit frontend."""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent_service import DEFAULT_AGENT_MODEL
from agent_tools import clear_agent_tool_cache
from api_client import AuroraApiClient, AuroraApiError
from settings_state import (
    AGENT_CHANNELS,
    AGENT_ORIGINS,
    PIPELINE_CHANNELS,
    PIPELINE_ORIGINS,
    RETRIEVAL_BACKENDS,
    init_agent_state,
    init_pipeline_state,
    new_run_id,
)

st.set_page_config(page_title="Settings · AURORA", layout="centered")


def _index(options: list[str], value: Any) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _check_health(base_url: str) -> dict[str, str]:
    client = AuroraApiClient(base_url)
    try:
        result = client.health()
    except AuroraApiError as exc:
        return {"kind": "error", "text": str(exc)}
    finally:
        client.close()
    return {
        "kind": "success",
        "text": f"{result.get('service', 'AURORA')} is {result.get('status', 'ok')}.",
    }


def _show_health(message: dict[str, str] | None) -> None:
    if not message:
        return
    if message["kind"] == "success":
        st.success(message["text"])
    else:
        st.error(message["text"])


init_pipeline_state()
init_agent_state(DEFAULT_AGENT_MODEL)

st.title("Settings")

pipeline_tab, agent_tab = st.tabs(["Pipeline", "AI Agent"])

with pipeline_tab:
    with st.form("pipeline_settings_form"):
        st.subheader("Server")
        api_base_url = st.text_input(
            "Pipeline API base URL",
            value=st.session_state["api_base_url"],
        )

        st.subheader("Run options")
        retrieval_backend = st.selectbox(
            "Pipeline retrieval backend",
            RETRIEVAL_BACKENDS,
            index=_index(RETRIEVAL_BACKENDS, st.session_state["retrieval_backend"]),
        )
        retrieval_k = st.number_input(
            "Pipeline snippets",
            min_value=1,
            max_value=20,
            value=int(st.session_state["retrieval_k"]),
            step=1,
        )
        channel = st.selectbox(
            "Pipeline channel",
            PIPELINE_CHANNELS,
            index=_index(PIPELINE_CHANNELS, st.session_state["channel"]),
        )
        origin = st.selectbox(
            "Pipeline origin",
            PIPELINE_ORIGINS,
            index=_index(PIPELINE_ORIGINS, st.session_state["origin"]),
        )
        strict_mode = st.toggle(
            "Pipeline strict evaluation mode",
            value=bool(st.session_state["strict_mode"]),
        )
        stage_backgrounds = st.toggle(
            "Tint stage bubbles",
            value=bool(st.session_state["stage_backgrounds"]),
        )
        pipeline_saved = st.form_submit_button("Save pipeline settings", type="primary")

    if pipeline_saved:
        st.session_state["api_base_url"] = api_base_url.strip().rstrip("/")
        st.session_state["retrieval_backend"] = retrieval_backend
        st.session_state["retrieval_k"] = int(retrieval_k)
        st.session_state["channel"] = channel
        st.session_state["origin"] = origin
        st.session_state["strict_mode"] = strict_mode
        st.session_state["stage_backgrounds"] = stage_backgrounds
        st.success("Pipeline settings saved for this session.")

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Check server health", use_container_width=True):
            st.session_state["pipeline_health_message"] = _check_health(
                st.session_state["api_base_url"]
            )
    with action_cols[1]:
        if st.button(
            "Clear conversation",
            disabled=not st.session_state.messages,
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.session_state.pipeline_run_id = None
            st.success("Pipeline conversation cleared.")

    _show_health(st.session_state.get("pipeline_health_message"))

with agent_tab:
    with st.form("agent_settings_form"):
        st.subheader("Server")
        agent_api_base_url = st.text_input(
            "Agent API base URL",
            value=st.session_state["agent_api_base_url"],
        )
        agent_model = st.text_input(
            "Agent model",
            value=st.session_state["agent_model"],
        )

        st.subheader("Run options")
        agent_retrieval_backend = st.selectbox(
            "Agent retrieval backend",
            RETRIEVAL_BACKENDS,
            index=_index(RETRIEVAL_BACKENDS, st.session_state["agent_retrieval_backend"]),
            key="agent_settings_retrieval_backend",
        )
        agent_retrieval_k = st.number_input(
            "Agent snippets",
            min_value=1,
            max_value=20,
            value=int(st.session_state["agent_retrieval_k"]),
            step=1,
            key="agent_settings_retrieval_k",
        )
        agent_channel = st.selectbox(
            "Agent channel",
            AGENT_CHANNELS,
            index=_index(AGENT_CHANNELS, st.session_state["agent_channel"]),
            key="agent_settings_channel",
        )
        agent_origin = st.selectbox(
            "Agent origin",
            AGENT_ORIGINS,
            index=_index(AGENT_ORIGINS, st.session_state["agent_origin"]),
            key="agent_settings_origin",
        )
        agent_strict_mode = st.toggle(
            "Agent strict evaluation mode",
            value=bool(st.session_state["agent_strict_mode"]),
        )
        agent_saved = st.form_submit_button("Save agent settings", type="primary")

    if agent_saved:
        st.session_state["agent_api_base_url"] = agent_api_base_url.strip().rstrip("/")
        st.session_state["agent_model"] = agent_model.strip() or DEFAULT_AGENT_MODEL
        st.session_state["agent_retrieval_backend"] = agent_retrieval_backend
        st.session_state["agent_retrieval_k"] = int(agent_retrieval_k)
        st.session_state["agent_channel"] = agent_channel
        st.session_state["agent_origin"] = agent_origin
        st.session_state["agent_strict_mode"] = agent_strict_mode
        st.success("Agent settings saved for this session.")

    agent_action_cols = st.columns(2)
    with agent_action_cols[0]:
        if st.button("Check agent server health", use_container_width=True):
            st.session_state["agent_health_message"] = _check_health(
                st.session_state["agent_api_base_url"]
            )
    with agent_action_cols[1]:
        if st.button(
            "Clear agent conversation",
            disabled=not st.session_state.agent_messages,
            use_container_width=True,
        ):
            clear_agent_tool_cache(st.session_state.agent_run_id)
            st.session_state.agent_messages = []
            st.session_state.agent_input_items = []
            st.session_state.agent_run_id = new_run_id()
            st.success("Agent conversation cleared.")

    _show_health(st.session_state.get("agent_health_message"))
