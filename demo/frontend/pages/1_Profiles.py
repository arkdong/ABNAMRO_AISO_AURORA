"""Profile registry — view, add, and modify profiles."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend._io import delete_profile, save_profile  # noqa: E402
from profiles import load_all, load_by_id  # noqa: E402
from profiles.loader import (  # noqa: E402
    CANONICAL_INTENT_CODES,
    DomainExpertProfile,
    WorkflowProfile,
)

st.set_page_config(page_title="Profiles · AURORA", layout="wide")
st.title("Profiles")
st.caption("Profile registry — view, add, or modify.")


# ── helpers ───────────────────────────────────────────────────────────────


def _to_lines(items) -> str:
    return "\n".join(items)


def _from_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _all_ids() -> list[str]:
    return [p.id for p in load_all()]


def _render_validation(path: Path, errors: list[str], action: str) -> None:
    rel = path.relative_to(REPO_ROOT) if path.exists() or action == "deleted" else path
    if errors:
        st.warning(f"{action.capitalize()} `{rel}` — validation reported issues:")
        for e in errors:
            st.write(f"- {e}")
    else:
        st.success(f"{action.capitalize()} `{rel}`")


def _profile_form(prefix: str, category: str, defaults: dict, lock_id: bool) -> dict:
    """Render the category-aware widgets and return the collected values."""
    data: dict = {}

    data["id"] = st.text_input(
        "ID (snake_case)",
        value=defaults.get("id", ""),
        disabled=lock_id,
        key=f"{prefix}_id",
        help="Stable identifier. domain_expert profiles conventionally use the 'expert_' prefix.",
    )
    data["name"] = st.text_input("Name", value=defaults.get("name", ""), key=f"{prefix}_name")
    data["description"] = st.text_area(
        "Description", value=defaults.get("description", ""), key=f"{prefix}_description"
    )

    if category == "workflow":
        data["intent_codes"] = st.multiselect(
            "Intent codes",
            options=sorted(CANONICAL_INTENT_CODES),
            default=defaults.get("intent_codes", []),
            key=f"{prefix}_intents",
        )
        data["skills"] = _from_lines(
            st.text_area(
                "Skills (one per line)",
                value=_to_lines(defaults.get("skills", [])),
                key=f"{prefix}_skills",
            )
        )
        data["tools"] = _from_lines(
            st.text_area(
                "Tools (one per line)",
                value=_to_lines(defaults.get("tools", [])),
                key=f"{prefix}_tools",
            )
        )
        data["guardrails"] = _from_lines(
            st.text_area(
                "Guardrails (one per line)",
                value=_to_lines(defaults.get("guardrails", [])),
                key=f"{prefix}_guardrails",
            )
        )
        data["outputs"] = _from_lines(
            st.text_area(
                "Outputs (one per line)",
                value=_to_lines(defaults.get("outputs", [])),
                key=f"{prefix}_outputs",
            )
        )
    else:
        data["sector"] = st.text_input(
            "Sector", value=defaults.get("sector", ""), key=f"{prefix}_sector"
        )
        data["topic_keywords"] = _from_lines(
            st.text_area(
                "Topic keywords (one per line)",
                value=_to_lines(defaults.get("topic_keywords", [])),
                key=f"{prefix}_keywords",
            )
        )
        data["expertise_areas"] = _from_lines(
            st.text_area(
                "Expertise areas (one per line)",
                value=_to_lines(defaults.get("expertise_areas", [])),
                key=f"{prefix}_expertise",
            )
        )
        data["style_signature"] = _from_lines(
            st.text_area(
                "Style signature (one per line)",
                value=_to_lines(defaults.get("style_signature", [])),
                key=f"{prefix}_style",
            )
        )

    data["knowledge"] = _from_lines(
        st.text_area(
            "Knowledge (one per line)",
            value=_to_lines(defaults.get("knowledge", [])),
            key=f"{prefix}_knowledge",
        )
    )

    other_ids = [pid for pid in _all_ids() if pid != defaults.get("id")]
    safe_default = [x for x in defaults.get("co_activates_with", []) if x in other_ids]
    data["co_activates_with"] = st.multiselect(
        "Co-activates with",
        options=other_ids,
        default=safe_default,
        key=f"{prefix}_coact",
    )

    return data


# ── tabs ──────────────────────────────────────────────────────────────────

tab_view, tab_add, tab_edit = st.tabs(["View", "Add", "Edit"])


# ── View ──────────────────────────────────────────────────────────────────

with tab_view:
    bundle = load_all()

    st.subheader(f"Workflow profiles ({len(bundle.workflow)})")
    for p in bundle.workflow:
        with st.expander(f"{p.name}  ·  `{p.id}`"):
            st.write(p.description)
            if p.activates_on_intent_codes:
                st.markdown(
                    "**Intent codes:** "
                    + ", ".join(f"`{c}`" for c in p.activates_on_intent_codes)
                )
            for label, items in [
                ("Skills", p.skills),
                ("Tools", p.tools),
                ("Guardrails", p.guardrails),
                ("Outputs", p.outputs),
                ("Knowledge", p.knowledge),
            ]:
                if items:
                    st.markdown(f"**{label}**")
                    for it in items:
                        st.write(f"- {it}")
            if p.co_activates_with:
                st.markdown(
                    "**Co-activates with:** "
                    + ", ".join(f"`{x}`" for x in p.co_activates_with)
                )

    st.subheader(f"Domain expert profiles ({len(bundle.domain_expert)})")
    for p in bundle.domain_expert:
        with st.expander(f"{p.name}  ·  `{p.id}`"):
            st.write(p.description)
            st.markdown(f"**Sector:** {p.sector}")
            if p.topic_keywords:
                st.markdown(
                    "**Topic keywords:** "
                    + ", ".join(f"`{k}`" for k in p.topic_keywords)
                )
            for label, items in [
                ("Expertise areas", p.expertise_areas),
                ("Style signature", p.style_signature),
                ("Knowledge", p.knowledge),
            ]:
                if items:
                    st.markdown(f"**{label}**")
                    for it in items:
                        st.write(f"- {it}")
            if p.co_activates_with:
                st.markdown(
                    "**Co-activates with:** "
                    + ", ".join(f"`{x}`" for x in p.co_activates_with)
                )


# ── Add ───────────────────────────────────────────────────────────────────

with tab_add:
    add_category = st.radio(
        "Category", ["workflow", "domain_expert"], key="add_category", horizontal=True
    )
    add_data = _profile_form("add", add_category, {}, lock_id=False)

    if st.button("Create profile", type="primary", key="add_submit"):
        if not add_data["id"]:
            st.error("ID is required.")
        elif add_data["id"] in _all_ids():
            st.error(f"ID '{add_data['id']}' already exists. Use the Edit tab.")
        elif not add_data["name"]:
            st.error("Name is required.")
        else:
            path, errors = save_profile(add_data, add_category)
            _render_validation(path, errors, "created")


# ── Edit ──────────────────────────────────────────────────────────────────

with tab_edit:
    ids = _all_ids()
    if not ids:
        st.info("No profiles yet.")
    else:
        selected = st.selectbox("Profile", options=ids, key="edit_select")
        p = load_by_id(selected)
        category = p.category
        st.caption(f"Category: `{category}` (locked)")

        if isinstance(p, WorkflowProfile):
            defaults = {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "intent_codes": list(p.activates_on_intent_codes),
                "skills": list(p.skills),
                "tools": list(p.tools),
                "guardrails": list(p.guardrails),
                "outputs": list(p.outputs),
                "knowledge": list(p.knowledge),
                "co_activates_with": list(p.co_activates_with),
            }
        else:
            assert isinstance(p, DomainExpertProfile)
            defaults = {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "sector": p.sector,
                "topic_keywords": list(p.topic_keywords),
                "expertise_areas": list(p.expertise_areas),
                "style_signature": list(p.style_signature),
                "knowledge": list(p.knowledge),
                "co_activates_with": list(p.co_activates_with),
            }

        edit_data = _profile_form(f"edit_{selected}", category, defaults, lock_id=True)
        edit_data["id"] = selected  # locked field

        col_save, col_delete = st.columns([1, 2])
        with col_save:
            save_clicked = st.button("Save", type="primary", key=f"edit_save_{selected}")
        with col_delete:
            confirm = st.checkbox(
                "Confirm delete", key=f"edit_confirm_{selected}",
                help="Required before the Delete button activates.",
            )
            delete_clicked = st.button(
                "Delete", key=f"edit_delete_{selected}", disabled=not confirm
            )

        if save_clicked:
            if not edit_data["name"]:
                st.error("Name is required.")
            else:
                path, errors = save_profile(edit_data, category)
                _render_validation(path, errors, "saved")

        if delete_clicked:
            path, errors = delete_profile(selected, category)
            _render_validation(path, errors, "deleted")
            st.info("Reload the page to refresh the profile list.")
