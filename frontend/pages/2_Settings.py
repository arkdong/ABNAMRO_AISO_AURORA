"""Settings page — Intent API key and model.

Values are stored in ``st.session_state`` and live for the duration of the
browser session (no on-disk persistence; this is a local POC).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Settings · AURORA", layout="centered")
st.title("Settings")
st.caption("Local POC — values are kept in session state, not on disk.")

st.session_state.setdefault("intent_api_key", "")
st.session_state.setdefault("intent_model", "gpt-4o-mini")

with st.form("settings_form"):
    api_key = st.text_input(
        "Intent API Key",
        value=st.session_state["intent_api_key"],
        type="password",
        help="OpenAI API key used by the intent classifier.",
    )
    model = st.text_input(
        "Model",
        value=st.session_state["intent_model"],
        help="OpenAI model name, e.g. gpt-4o-mini, gpt-4o, gpt-5-mini.",
    )
    saved = st.form_submit_button("Save", type="primary")

if saved:
    st.session_state["intent_api_key"] = api_key.strip()
    st.session_state["intent_model"] = model.strip()
    st.success("Settings saved for this session.")

st.divider()
key_set = bool(st.session_state["intent_api_key"])
model_set = bool(st.session_state["intent_model"])
if key_set and model_set:
    st.info(f"Intent classifier will use the LLM path (`{st.session_state['intent_model']}`).")
else:
    st.warning(
        "Intent classifier will use the **deterministic keyword fallback** "
        "until both an API key and a model are saved."
    )
