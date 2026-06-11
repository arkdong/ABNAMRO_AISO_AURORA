"""Settings page — API keys and models per pipeline stage.

Values are stored in ``st.session_state`` and live for the duration of the
browser session (no on-disk persistence; this is a local POC). Both stages
seed their defaults from ``.env`` so a developer can drop the keys in
``OPENAI_API_KEY_*`` and never have to retype them in the UI.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Settings · AURORA", layout="centered")
st.title("Settings")
st.caption("Local POC — values are kept in session state, not on disk.")

st.session_state.setdefault("intent_api_key", os.getenv("OPENAI_API_KEY_INTENT", ""))
st.session_state.setdefault("intent_model", "gpt-4o-mini")
st.session_state.setdefault(
    "content_api_key", os.getenv("OPENAI_API_KEY_CONTENT_GENERATION", "")
)
st.session_state.setdefault("content_model", "gpt-4o")
st.session_state.setdefault(
    "eval_api_key", os.getenv("OPENAI_API_KEY_EVALUATION", "")
)
st.session_state.setdefault("eval_model", "gpt-4o-mini")
st.session_state.setdefault("eval_strict_mode", False)
st.session_state.setdefault("stage_backgrounds", True)

with st.form("settings_form"):
    st.subheader("Intent / refinement")
    api_key = st.text_input(
        "Intent API Key",
        value=st.session_state["intent_api_key"],
        type="password",
        help=(
            "OpenAI API key used by the intent classifier and refinement "
            "generator. Defaults to OPENAI_API_KEY_INTENT from .env."
        ),
    )
    model = st.text_input(
        "Intent Model",
        value=st.session_state["intent_model"],
        help="OpenAI model name, e.g. gpt-4o-mini, gpt-4o, gpt-5-mini.",
    )

    st.subheader("Content generation (Stage 5)")
    content_api_key = st.text_input(
        "Content Generation API Key",
        value=st.session_state["content_api_key"],
        type="password",
        help=(
            "OpenAI API key used to write the final content from the refined "
            "prompt + retrieved snippets. Defaults to "
            "OPENAI_API_KEY_CONTENT_GENERATION from .env."
        ),
    )
    content_model = st.text_input(
        "Content Generation Model",
        value=st.session_state["content_model"],
        help="OpenAI model for the final write-up (gpt-4o is a reasonable default).",
    )

    st.subheader("Evaluation (Stage 6)")
    eval_api_key = st.text_input(
        "Evaluation API Key",
        value=st.session_state["eval_api_key"],
        type="password",
        help=(
            "OpenAI API key used by the Tier 2 LLM-judge rubrics that score "
            "generated content against the KPI catalogue. Defaults to "
            "OPENAI_API_KEY_EVALUATION from .env. Tier 1 deterministic "
            "checks always run regardless of this key."
        ),
    )
    eval_model = st.text_input(
        "Evaluation Model",
        value=st.session_state["eval_model"],
        help=(
            "OpenAI model for the judge calls. gpt-4o-mini is plenty for "
            "short rubric scoring and keeps eval cost low."
        ),
    )
    eval_strict = st.toggle(
        "Strict mode",
        value=st.session_state["eval_strict_mode"],
        help=(
            "When ON, Tier 2 KPIs without a working LLM judge are marked "
            "as failing instead of passing. Recommended for production-like "
            "runs. OFF (default) keeps dev runs flowing without a key."
        ),
    )

    st.subheader("UI")
    stage_bg = st.toggle(
        "Tint stage bubbles",
        value=st.session_state["stage_backgrounds"],
        help=(
            "Tint each pipeline stage's chat bubble with a light background "
            "colour so the stages are easier to scan."
        ),
    )
    saved = st.form_submit_button("Save", type="primary")

if saved:
    st.session_state["intent_api_key"] = api_key.strip()
    st.session_state["intent_model"] = model.strip()
    st.session_state["content_api_key"] = content_api_key.strip()
    st.session_state["content_model"] = content_model.strip()
    st.session_state["eval_api_key"] = eval_api_key.strip()
    st.session_state["eval_model"] = eval_model.strip()
    st.session_state["eval_strict_mode"] = eval_strict
    st.session_state["stage_backgrounds"] = stage_bg
    st.success("Settings saved for this session.")

st.divider()
intent_set = bool(st.session_state["intent_api_key"]) and bool(
    st.session_state["intent_model"]
)
if intent_set:
    st.info(
        f"Intent classifier will use the LLM path "
        f"(`{st.session_state['intent_model']}`)."
    )
else:
    st.warning(
        "Intent classifier will use the **deterministic keyword fallback** "
        "until both an API key and a model are saved."
    )

content_set = bool(st.session_state["content_api_key"]) and bool(
    st.session_state["content_model"]
)
if content_set:
    st.info(
        f"Content generation will use `{st.session_state['content_model']}`."
    )
else:
    st.warning(
        "Content generation will return a **stub placeholder** until both an "
        "API key and a model are saved."
    )

eval_set = bool(st.session_state["eval_api_key"]) and bool(
    st.session_state["eval_model"]
)
if eval_set:
    st.info(
        f"Evaluation will run Tier 1 + Tier 2 LLM judges via "
        f"`{st.session_state['eval_model']}`."
        + (" Strict mode: ON." if st.session_state["eval_strict_mode"] else "")
    )
else:
    msg = (
        "Evaluation will run **Tier 1 deterministic checks only** until both "
        "an API key and a model are saved."
    )
    if st.session_state["eval_strict_mode"]:
        st.error(
            msg
            + " Strict mode is **ON**, so all Tier 2 KPIs will be marked failing."
        )
    else:
        st.warning(msg)
