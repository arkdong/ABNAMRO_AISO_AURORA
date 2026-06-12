"""Shared Streamlit session defaults for AURORA frontend pages."""

from __future__ import annotations

import os
import uuid

import streamlit as st

RETRIEVAL_BACKENDS = ["pageindex", "vector_rag"]
PIPELINE_CHANNELS = ["web", "chat", "messages", "employee", "app_ib"]
PIPELINE_ORIGINS = ["instant", "human", "genai_knowledge"]
AGENT_CHANNELS = ["web", "chat", "messages", "employee", "app_ib"]
AGENT_ORIGINS = ["human", "genai_knowledge", "instant"]


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def init_pipeline_state() -> None:
    st.session_state.setdefault(
        "api_base_url",
        os.getenv("AURORA_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retrieval_backend", "pageindex")
    st.session_state.setdefault("retrieval_k", 5)
    st.session_state.setdefault("channel", "web")
    st.session_state.setdefault("origin", "instant")
    st.session_state.setdefault("strict_mode", False)
    st.session_state.setdefault("stage_backgrounds", True)
    st.session_state.setdefault("pipeline_run_id", None)


def init_agent_state(default_model: str = "gpt-5-mini") -> None:
    st.session_state.setdefault(
        "agent_api_base_url",
        os.getenv("AURORA_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
    )
    st.session_state.setdefault("agent_model", default_model)
    st.session_state.setdefault("agent_retrieval_backend", "pageindex")
    st.session_state.setdefault("agent_retrieval_k", 5)
    st.session_state.setdefault("agent_channel", "web")
    st.session_state.setdefault("agent_origin", "instant")
    st.session_state.setdefault("agent_strict_mode", False)
    st.session_state.setdefault("agent_run_id", new_run_id())
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("agent_input_items", [])
    st.session_state.setdefault("agent_health_message", None)
