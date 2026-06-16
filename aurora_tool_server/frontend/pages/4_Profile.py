"""Profile management page for the AURORA Streamlit frontend."""

from __future__ import annotations

from typing import Any, Literal

import streamlit as st

from api_client import AuroraApiClient, AuroraApiError
from branding import apply_branding
from settings_state import init_pipeline_state

ProfileCategory = Literal["workflow", "domain_expert"]

CATEGORY_LABELS: dict[ProfileCategory, str] = {
    "workflow": "Workflow",
    "domain_expert": "Domain experts",
}


st.set_page_config(page_title="Profile · AURORA", layout="wide")
init_pipeline_state()
apply_branding("Profile", "Manage AURORA profile assets")


def _lines(values: list[str] | None) -> str:
    return "\n".join(values or [])


def _parse_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _profile_options(profiles: list[dict[str, Any]]) -> list[str]:
    return [str(profile["id"]) for profile in profiles]


def _profile_by_id(
    profiles: list[dict[str, Any]],
    profile_id: str,
) -> dict[str, Any] | None:
    for profile in profiles:
        if profile.get("id") == profile_id:
            return profile
    return None


def _load_profiles() -> dict[str, Any]:
    client = AuroraApiClient(st.session_state["api_base_url"])
    try:
        return client.list_profiles()
    finally:
        client.close()


def _show_notice() -> None:
    notice = st.session_state.pop("profile_notice", None)
    if notice:
        st.success(notice)


def _submit_create(profile: dict[str, Any]) -> None:
    client = AuroraApiClient(st.session_state["api_base_url"])
    try:
        created = client.create_profile(profile)
    except AuroraApiError as exc:
        st.error(str(exc))
        return
    finally:
        client.close()
    st.session_state["profile_notice"] = f"Created {created['name']}."
    st.rerun()


def _submit_update(category: ProfileCategory, profile: dict[str, Any]) -> None:
    profile_id = str(profile["id"])
    client = AuroraApiClient(st.session_state["api_base_url"])
    try:
        updated = client.update_profile(category, profile_id, profile)
    except AuroraApiError as exc:
        st.error(str(exc))
        return
    finally:
        client.close()
    st.session_state["profile_notice"] = f"Updated {updated['name']}."
    st.rerun()


def _submit_delete(category: ProfileCategory, profile: dict[str, Any]) -> None:
    client = AuroraApiClient(st.session_state["api_base_url"])
    try:
        deleted = client.delete_profile(category, str(profile["id"]))
    except AuroraApiError as exc:
        st.error(str(exc))
        return
    finally:
        client.close()
    st.session_state["profile_notice"] = f"Deleted {deleted['name']}."
    st.rerun()


def _profile_payload(
    *,
    form_key: str,
    category: ProfileCategory,
    initial: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    is_edit = initial is not None
    data = initial or {}
    with st.form(form_key):
        left, right = st.columns(2)
        with left:
            profile_id = st.text_input(
                "Profile ID",
                value=str(data.get("id") or ""),
                disabled=is_edit,
                key=f"{form_key}_id",
            )
            name = st.text_input(
                "Name",
                value=str(data.get("name") or ""),
                key=f"{form_key}_name",
            )
        with right:
            description = st.text_area(
                "Description",
                value=str(data.get("description") or ""),
                height=110,
                key=f"{form_key}_description",
            )

        if category == "workflow":
            intent_codes = st.text_area(
                "Intent codes",
                value=_lines(data.get("activates_on_intent_codes")),
                height=90,
                key=f"{form_key}_intent_codes",
            )
            skills = st.text_area(
                "Skills",
                value=_lines(data.get("skills")),
                height=120,
                key=f"{form_key}_skills",
            )
            tools = st.text_area(
                "Tools",
                value=_lines(data.get("tools")),
                height=90,
                key=f"{form_key}_tools",
            )
            guardrails = st.text_area(
                "Guardrails",
                value=_lines(data.get("guardrails")),
                height=120,
                key=f"{form_key}_guardrails",
            )
            outputs = st.text_area(
                "Outputs",
                value=_lines(data.get("outputs")),
                height=100,
                key=f"{form_key}_outputs",
            )
            sector = None
            topic_keywords = ""
            expertise_areas = ""
            style_signature = ""
        else:
            sector = st.text_input(
                "Sector",
                value=str(data.get("sector") or ""),
                key=f"{form_key}_sector",
            )
            topic_keywords = st.text_area(
                "Topic keywords",
                value=_lines(data.get("topic_keywords")),
                height=110,
                key=f"{form_key}_topic_keywords",
            )
            expertise_areas = st.text_area(
                "Expertise areas",
                value=_lines(data.get("expertise_areas")),
                height=130,
                key=f"{form_key}_expertise_areas",
            )
            style_signature = st.text_area(
                "Style signature",
                value=_lines(data.get("style_signature")),
                height=110,
                key=f"{form_key}_style_signature",
            )
            intent_codes = ""
            skills = ""
            tools = ""
            guardrails = ""
            outputs = ""

        knowledge = st.text_area(
            "Knowledge",
            value=_lines(data.get("knowledge")),
            height=120,
            key=f"{form_key}_knowledge",
        )
        co_activates_with = st.text_area(
            "Co-activates with",
            value=_lines(data.get("co_activates_with")),
            height=90,
            key=f"{form_key}_co_activates_with",
        )

        submitted = st.form_submit_button(
            "Save profile" if is_edit else "Add profile",
            type="primary",
        )

    payload = {
        "id": str(data.get("id") if is_edit else profile_id).strip(),
        "name": name.strip(),
        "description": description.strip(),
        "category": category,
        "activates_on_intent_codes": _parse_lines(intent_codes),
        "sector": sector.strip() if isinstance(sector, str) else None,
        "topic_keywords": _parse_lines(topic_keywords),
        "knowledge": _parse_lines(knowledge),
        "skills": _parse_lines(skills),
        "tools": _parse_lines(tools),
        "guardrails": _parse_lines(guardrails),
        "outputs": _parse_lines(outputs),
        "expertise_areas": _parse_lines(expertise_areas),
        "style_signature": _parse_lines(style_signature),
        "co_activates_with": _parse_lines(co_activates_with),
    }
    return submitted, payload


def _render_delete(category: ProfileCategory, profile: dict[str, Any]) -> None:
    profile_id = str(profile["id"])
    confirm_key = f"profile_delete_confirm_{category}_{profile_id}"
    st.divider()
    confirmed = st.checkbox(
        f"Confirm delete {profile_id}",
        key=confirm_key,
    )
    if st.button(
        "Delete profile",
        disabled=not confirmed,
        key=f"profile_delete_{category}_{profile_id}",
        use_container_width=True,
    ):
        _submit_delete(category, profile)


def _render_category(
    category: ProfileCategory,
    profiles: list[dict[str, Any]],
) -> None:
    label = CATEGORY_LABELS[category]
    st.subheader(label)
    edit_tab, add_tab = st.tabs(["Edit", "Add"])

    with edit_tab:
        if not profiles:
            st.info(f"No {label.lower()} profiles found.")
        else:
            options = _profile_options(profiles)
            selected_id = st.selectbox(
                "Profile",
                options=options,
                format_func=lambda profile_id: (
                    f"{_profile_by_id(profiles, profile_id)['name']} ({profile_id})"
                    if _profile_by_id(profiles, profile_id)
                    else profile_id
                ),
                key=f"profile_select_{category}",
            )
            selected = _profile_by_id(profiles, selected_id)
            if selected:
                submitted, payload = _profile_payload(
                    form_key=f"profile_edit_{category}_{selected_id}",
                    category=category,
                    initial=selected,
                )
                if submitted:
                    _submit_update(category, payload)
                _render_delete(category, selected)

    with add_tab:
        submitted, payload = _profile_payload(
            form_key=f"profile_add_{category}",
            category=category,
        )
        if submitted:
            _submit_create(payload)


_show_notice()
st.caption(f"API base URL: {st.session_state['api_base_url']}")

try:
    catalogue = _load_profiles()
except AuroraApiError as exc:
    st.error(str(exc))
    st.stop()

workflow_tab, expert_tab = st.tabs(["Workflow", "Domain experts"])
with workflow_tab:
    _render_category("workflow", catalogue.get("workflow") or [])
with expert_tab:
    _render_category("domain_expert", catalogue.get("domain_expert") or [])
