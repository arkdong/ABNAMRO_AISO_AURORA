"""AURORA — Streamlit entry point.

Chat input on the home page. Each user request is run through the intent
classifier wrapper (frontend/_intent.py) which delegates to the teammate's
classifier in task_definition/. API key and model are configured on the
Settings page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend._intent import classify  # noqa: E402
from config import TASK_LABELS  # noqa: E402

st.set_page_config(page_title="AURORA", layout="centered")
st.title("AURORA")

st.session_state.setdefault("intent_api_key", "")
st.session_state.setdefault("intent_model", "gpt-4o-mini")
st.session_state.setdefault("messages", [])

key_set = bool(st.session_state["intent_api_key"])
model_set = bool(st.session_state["intent_model"])
if key_set and model_set:
    st.caption(f"Intent classifier: LLM · `{st.session_state['intent_model']}`")
else:
    st.caption("Intent classifier: deterministic fallback — set the Intent API Key on **Settings**.")


def _render_classification(role, task_code, confidence, reason, source) -> str:
    label = TASK_LABELS.get(task_code, task_code)
    return (
        f"**Intent classification** _(via {source})_\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        f"| Role | {role} |\n"
        f"| Task | `{task_code}` — {label} |\n"
        f"| Confidence | {confidence:.2f} |\n"
        f"| Reason | {reason} |\n"
    )


with st.sidebar:
    if st.button("Clear conversation", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("raw"):
            with st.expander("Raw LLM output"):
                st.code(m["raw"], language="json")

prompt = st.chat_input("Ask AURORA…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    role, task_code, confidence, reason, raw, source = classify(
        prompt,
        api_key=st.session_state["intent_api_key"] or None,
        model=st.session_state["intent_model"] or None,
    )
    reply = _render_classification(role, task_code, confidence, reason, source)
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "raw": raw}
    )
    with st.chat_message("assistant"):
        st.markdown(reply)
        if raw:
            with st.expander("Raw LLM output"):
                st.code(raw, language="json")
