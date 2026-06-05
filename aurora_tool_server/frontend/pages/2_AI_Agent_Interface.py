"""OpenAI Agents SDK interface for AURORA."""

from __future__ import annotations

import os

import streamlit as st

from agent_service import (
    DEFAULT_AGENT_MODEL,
    AuroraAgentError,
    AuroraAgentSettings,
    readiness_error,
    run_agent_turn,
)
from api_client import AuroraApiClient, AuroraApiError


st.set_page_config(page_title="AI Agent Interface · AURORA", layout="centered")
st.title("AI Agent Interface")


def _init_state() -> None:
    st.session_state.setdefault(
        "agent_api_base_url",
        os.getenv("AURORA_API_BASE_URL", "http://127.0.0.1:8000"),
    )
    st.session_state.setdefault("agent_model", DEFAULT_AGENT_MODEL)
    st.session_state.setdefault("agent_retrieval_backend", "pageindex")
    st.session_state.setdefault("agent_retrieval_k", 5)
    st.session_state.setdefault("agent_channel", "web")
    st.session_state.setdefault("agent_origin", "instant")
    st.session_state.setdefault("agent_strict_mode", False)
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("agent_input_items", [])
    st.session_state.setdefault("agent_health_message", None)


def _settings() -> AuroraAgentSettings:
    return AuroraAgentSettings.from_values(
        api_base_url=st.session_state["agent_api_base_url"],
        model=st.session_state["agent_model"],
        retrieval_backend=st.session_state["agent_retrieval_backend"],
        k=int(st.session_state["agent_retrieval_k"]),
        channel=st.session_state["agent_channel"],
        origin=st.session_state["agent_origin"],
        strict_mode=bool(st.session_state["agent_strict_mode"]),
    )


def _check_health() -> None:
    client = AuroraApiClient(st.session_state["agent_api_base_url"])
    try:
        result = client.health()
        st.session_state.agent_health_message = {
            "kind": "success",
            "text": f"{result.get('service', 'AURORA')} is {result.get('status', 'ok')}.",
        }
    except AuroraApiError as exc:
        st.session_state.agent_health_message = {"kind": "error", "text": str(exc)}
    finally:
        client.close()


def _clear_conversation() -> None:
    st.session_state.agent_messages = []
    st.session_state.agent_input_items = []


_init_state()

with st.sidebar:
    st.text_input("API base URL", key="agent_api_base_url")
    st.text_input("Agent model", key="agent_model")
    st.selectbox(
        "Retrieval backend",
        ["pageindex", "vector_rag"],
        key="agent_retrieval_backend",
    )
    st.slider("Snippet count", 1, 20, key="agent_retrieval_k")
    st.selectbox("Channel", ["web", "chat", "messages", "employee", "app_ib"], key="agent_channel")
    st.selectbox("Origin", ["human", "genai_knowledge", "instant"], key="agent_origin")
    st.checkbox("Strict evaluation mode", key="agent_strict_mode")
    st.button("Check server health", on_click=_check_health)
    if st.session_state.agent_health_message:
        if st.session_state.agent_health_message["kind"] == "success":
            st.success(st.session_state.agent_health_message["text"])
        else:
            st.error(st.session_state.agent_health_message["text"])
    st.button("Clear agent conversation", on_click=_clear_conversation)

setup_error = readiness_error()
if setup_error:
    st.warning(setup_error)

for message in st.session_state.agent_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])

prompt = st.chat_input("Ask the AURORA agent", disabled=setup_error is not None)

if prompt:
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running AURORA agent..."):
            try:
                result = run_agent_turn(
                    prompt,
                    settings=_settings(),
                    input_items=st.session_state.agent_input_items,
                )
            except AuroraAgentError as exc:
                content = f"Agent error: {exc}"
                st.error(content)
                st.session_state.agent_messages.append(
                    {"role": "assistant", "content": content}
                )
            else:
                st.markdown(result.final_output)
                caption = f"Answered by {result.last_agent_name}"
                st.caption(caption)
                st.session_state.agent_input_items = result.input_items
                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": result.final_output,
                        "caption": caption,
                    }
                )
