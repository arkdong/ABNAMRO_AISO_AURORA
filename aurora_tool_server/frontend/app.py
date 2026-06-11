"""Server-backed AURORA Streamlit demo."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from api_client import AuroraApiClient, AuroraApiError

TASK_LABELS = {
    "T1_DRAFT": "Draft new content",
    "T1_TRANSLATE": "Translate existing content",
    "T1_SEARCH": "Search corpus for related articles",
    "T2_COMPLIANCE": "Quality and compliance check",
    "T4_RENEWAL": "Detect and renew aging articles",
}

MAX_REFINEMENT_TURNS = 5

STAGE_STYLES: dict[str, tuple[str, str, str, str]] = {
    "intent": ("I", "#dbeafe", "#1e40af", "Stage 1 · Intent"),
    "profiles": ("P", "#dcfce7", "#166534", "Stage 2 · Profiles"),
    "retrieval": ("R", "#ede9fe", "#5b21b6", "Stage 3 · Retrieval"),
    "refinement": ("Q", "#fef3c7", "#92400e", "Stage 4 · Refinement"),
    "content": ("C", "#e2e8f0", "#0f172a", "Stage 5 · Content"),
    "evaluation": ("E", "#cffafe", "#155e75", "Stage 6 · Evaluation"),
    "error": ("!", "#fee2e2", "#991b1b", "Server error"),
}

TIMELINE_LABELS = {
    "user": "Prompt",
    "intent": "Intent",
    "profiles": "Profiles",
    "retrieval": "Retrieval",
    "refinement": "Refinement",
    "content": "Content",
    "evaluation": "Evaluation",
    "error": "Error",
}


st.set_page_config(page_title="AURORA", layout="centered")
st.title("AURORA")


def _init_state() -> None:
    st.session_state.setdefault(
        "api_base_url",
        os.getenv("AURORA_API_BASE_URL", "http://127.0.0.1:8000"),
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retrieval_backend", "pageindex")
    st.session_state.setdefault("retrieval_k", 5)
    st.session_state.setdefault("channel", "web")
    st.session_state.setdefault("origin", "instant")
    st.session_state.setdefault("strict_mode", False)
    st.session_state.setdefault("stage_backgrounds", True)


_init_state()


def _options() -> dict[str, Any]:
    return {
        "k": int(st.session_state["retrieval_k"]),
        "retrieval_backend": st.session_state["retrieval_backend"],
        "channel": st.session_state["channel"],
        "origin": st.session_state["origin"],
        "strict_mode": bool(st.session_state["strict_mode"]),
    }


def _with_client(fn):
    client = AuroraApiClient(st.session_state["api_base_url"])
    try:
        return fn(client)
    finally:
        client.close()


def _append_error(stage: str, message: str, user_prompt: str | None = None) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "error",
            "stage": stage,
            "message": message,
            "user_prompt": user_prompt,
        }
    )


def _stage_chip(kind: str) -> str:
    _, bg, fg, label = STAGE_STYLES[kind]
    return (
        f'<span data-stage-chip="{kind}" '
        f'style="background:{bg};color:{fg};padding:3px 12px;'
        f'border-radius:10px;font-weight:600;font-size:0.82em;">'
        f"{label}</span>"
    )


def _block_anchor(kind: str, idx: int) -> str:
    return (
        f'<span id="block-{kind}-{idx}" data-block-anchor="block-{kind}-{idx}" '
        f'style="display:block;height:0;scroll-margin-top:5rem;"></span>'
    )


def _substep_chip(label: str, kind: str = "question") -> str:
    return (
        f'<span data-substep-chip="{kind}" '
        f'style="background:#fff7ed;color:#9a3412;padding:1px 8px;'
        f'border-radius:6px;font-weight:500;font-size:0.72em;">{label}</span>'
    )


def _inject_stage_styles() -> None:
    if not st.session_state.get("stage_backgrounds", True):
        return
    st.markdown(
        """
        <style>
          [data-testid="stChatMessage"]:is(
            :has([data-stage-chip="intent"]),
            :has([data-stage-chip="profiles"]),
            :has([data-stage-chip="retrieval"]),
            :has([data-stage-chip="refinement"]),
            :has([data-stage-chip="content"]),
            :has([data-stage-chip="evaluation"]),
            :has([data-stage-chip="error"]),
            :has([data-substep-chip="question"]),
            :has([data-substep-chip="footer"])
          ) {
            padding: 0.9rem 1.25rem 0.9rem 1rem;
            margin-right: 0.75rem;
            box-sizing: border-box;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="intent"]) {
            background: linear-gradient(135deg, #dbeafe40, #dbeafe10);
            border-left: 4px solid #1e40af;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="profiles"]) {
            background: linear-gradient(135deg, #dcfce740, #dcfce710);
            border-left: 4px solid #166534;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="retrieval"]) {
            background: linear-gradient(135deg, #ede9fe40, #ede9fe10);
            border-left: 4px solid #5b21b6;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="refinement"]) {
            background: linear-gradient(135deg, #fef3c740, #fef3c710);
            border-left: 4px solid #92400e;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="content"]) {
            background: linear-gradient(135deg, #e2e8f040, #e2e8f010);
            border-left: 4px solid #0f172a;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="evaluation"]) {
            background: linear-gradient(135deg, #cffafe40, #cffafe10);
            border-left: 4px solid #155e75;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-stage-chip="error"]) {
            background: #fee2e255;
            border-left: 4px solid #991b1b;
            border-radius: 10px;
          }
          [data-testid="stChatMessage"]:has([data-substep-chip="question"]) {
            background: #fff7ed55;
            border-left: 3px solid #c2410c;
            border-radius: 8px;
            margin-left: 3rem;
            max-width: calc(100% - 3rem - 0.75rem);
          }
          [data-testid="stChatMessage"]:has([data-substep-chip="footer"]) {
            background: #fef9c355;
            border-left: 3px solid #ca8a04;
            border-radius: 8px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_stage_styles()


def _intent_markdown(intent: dict[str, Any]) -> str:
    task_cells = ", ".join(
        f"`{code}` — {TASK_LABELS.get(code, code)}" for code in intent.get("task_codes", [])
    )
    rows = [
        f"| Role | {intent.get('role', '')} |",
        f"| Tasks | {task_cells} |",
        f"| Confidence | {float(intent.get('confidence', 0)):.2f} |",
        f"| Reason | {intent.get('task_reason', '')} |",
    ]
    if intent.get("sector"):
        rows.append(f"| Sector | {intent['sector']} |")
    if intent.get("topic_keywords"):
        rows.append(
            "| Topic keywords | "
            + ", ".join(f"`{kw}`" for kw in intent.get("topic_keywords", []))
            + " |"
        )
    if intent.get("language"):
        rows.append(f"| Language | {intent['language']} |")
    source = intent.get("source", "deterministic")
    return (
        f"**Intent classification** _(via {source})_\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        + "\n".join(rows)
    )


def _profiles_markdown(intent: dict[str, Any], bundle: dict[str, Any]) -> str:
    workflow = bundle.get("workflow", [])
    experts = bundle.get("domain_expert", [])
    if not workflow and not experts:
        return "_No profiles matched._"

    source = bundle.get("source", "deterministic")
    parts = [f"**Filtered profiles** _(via {source})_\n"]
    if bundle.get("reasoning"):
        parts.append(f"_Reasoning: {bundle['reasoning']}_\n")
    parts.append(f"**Workflow** ({len(workflow)})")
    if workflow:
        intent_codes = set(intent.get("task_codes", []))
        for profile in workflow:
            activated = [
                code
                for code in profile.get("activates_on_intent_codes", [])
                if code in intent_codes
            ]
            tags = ", ".join(f"`{code}`" for code in activated) or "-"
            reason = profile.get("selection_reason")
            suffix = f"  \n  Reason: {reason}" if reason else ""
            parts.append(
                f"- **{profile['name']}** (`{profile['id']}`) — activated by {tags}{suffix}"
            )
    else:
        parts.append("- _none matched_")

    parts.append(f"\n**Domain experts** ({len(experts)})")
    if experts:
        keywords = {kw.lower() for kw in intent.get("topic_keywords", [])}
        for profile in experts:
            matched = [
                kw for kw in profile.get("topic_keywords", []) if kw.lower() in keywords
            ]
            matched_text = ", ".join(f"`{kw}`" for kw in matched) if matched else "_sector match_"
            parts.append(
                f"- **{profile['name']}** (`{profile['id']}`)  \n"
                f"  Sector: {profile.get('sector') or '-'}  \n"
                f"  Matched keywords: {matched_text}"
                + (f"  \n  Reason: {profile['selection_reason']}" if profile.get("selection_reason") else "")
            )
    else:
        parts.append("- _none matched_")
    return "\n".join(parts)


def _retrieval_markdown(result: dict[str, Any]) -> str:
    query = result.get("query", {})
    corpora = ", ".join(f"`{c}`" for c in result.get("corpora_searched", [])) or "-"
    model = f" · `{result.get('model')}`" if result.get("model") else ""
    reasoning = f"  \n_Reasoning: {result['reasoning']}_" if result.get("reasoning") else ""
    return (
        f"**Context retrieval** _(via {result.get('provider')} · {result.get('source')}{model})_  \n"
        f"Corpora searched: {corpora}  \n"
        f"Query: k={query.get('k')}, task_codes=`{query.get('task_codes')}`, "
        f"sector=`{query.get('sector')}`  \n"
        f"Snippets returned: **{len(result.get('snippets', []))}**"
        f"{reasoning}"
    )


def _render_profile_expander(profile: dict[str, Any]) -> None:
    with st.expander(f"{profile.get('name')} (`{profile.get('id')}`) — details"):
        st.markdown(f"**Description:** {profile.get('description', '')}")
        if profile.get("sector"):
            st.markdown(f"**Sector:** {profile['sector']}")
        for label, key in [
            ("Intent codes", "activates_on_intent_codes"),
            ("Topic keywords", "topic_keywords"),
            ("Skills", "skills"),
            ("Expertise areas", "expertise_areas"),
            ("Knowledge", "knowledge"),
            ("Guardrails", "guardrails"),
            ("Outputs", "outputs"),
        ]:
            items = profile.get(key) or []
            if items:
                st.markdown(f"**{label}:**")
                for item in items:
                    st.markdown(f"- {item}")


def _render_snippet_expander(index: int, snippet: dict[str, Any]) -> None:
    label = (
        f"#{index + 1} · {snippet.get('title')} · `{snippet.get('source_doc')}` · "
        f"score={float(snippet.get('score', 0)):.2f}"
    )
    with st.expander(label):
        st.markdown(f"**Reason:** {snippet.get('reason', '')}")
        if snippet.get("line_num") is not None:
            st.markdown(
                f"**Location:** node `{snippet.get('node_id')}`, line/page {snippet.get('line_num')}"
            )
        if snippet.get("source_url"):
            st.markdown(f"**Source URL:** {snippet['source_url']}")
        body = snippet.get("content", "")
        if len(body) > 2000:
            body = body[:2000].rstrip() + "\n\n_(truncated)_"
        st.markdown("**Content:**")
        st.markdown(body)


def _render_intent_message(idx: int, message: dict[str, Any], is_latest: bool) -> None:
    with st.chat_message("assistant"):
        st.markdown(_block_anchor("intent", idx) + _stage_chip("intent"), unsafe_allow_html=True)
        st.markdown(_intent_markdown(message["intent"]))
        if is_latest and not message.get("proceeded"):
            if st.button("Proceed -> Filter profiles", key=f"profiles_{idx}", type="primary"):
                try:
                    with st.spinner("Filtering profiles via AURORA API..."):
                        bundle = _with_client(
                            lambda c: c.select_profiles(message["intent"], _options())
                        )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "kind": "profiles",
                            "intent": message["intent"],
                            "bundle": bundle,
                            "user_prompt": message["user_prompt"],
                            "proceeded": False,
                        }
                    )
                    message["proceeded"] = True
                except AuroraApiError as exc:
                    _append_error("profiles", str(exc), message.get("user_prompt"))
                st.rerun()


def _render_profiles_message(idx: int, message: dict[str, Any], is_latest: bool) -> None:
    bundle = message["bundle"]
    with st.chat_message("assistant"):
        st.markdown(
            _block_anchor("profiles", idx) + _stage_chip("profiles"),
            unsafe_allow_html=True,
        )
        st.markdown(_profiles_markdown(message["intent"], bundle))
        for profile in bundle.get("workflow", []):
            _render_profile_expander(profile)
        for profile in bundle.get("domain_expert", []):
            _render_profile_expander(profile)
        if is_latest and not message.get("proceeded"):
            if st.button("Proceed -> Retrieve context", key=f"retrieve_{idx}", type="primary"):
                try:
                    with st.spinner("Retrieving context via AURORA API..."):
                        result = _with_client(
                            lambda c: c.retrieve_context(
                                message["user_prompt"],
                                message["intent"],
                                bundle,
                                _options(),
                            )
                        )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "kind": "retrieval",
                            "intent": message["intent"],
                            "bundle": bundle,
                            "result": result,
                            "user_prompt": message["user_prompt"],
                            "proceeded": False,
                        }
                    )
                    message["proceeded"] = True
                except AuroraApiError as exc:
                    _append_error("retrieval", str(exc), message.get("user_prompt"))
                st.rerun()


def _render_retrieval_message(idx: int, message: dict[str, Any], is_latest: bool) -> None:
    result = message["result"]
    with st.chat_message("assistant"):
        st.markdown(
            _block_anchor("retrieval", idx) + _stage_chip("retrieval"),
            unsafe_allow_html=True,
        )
        st.markdown(_retrieval_markdown(result))
        if result.get("snippets"):
            for snippet_index, snippet in enumerate(result["snippets"]):
                _render_snippet_expander(snippet_index, snippet)
        else:
            st.markdown("_No snippets found._")
        if is_latest and not message.get("proceeded"):
            if st.button("Proceed -> Refine prompt", key=f"refine_{idx}", type="primary"):
                try:
                    with st.spinner("Preparing refinement questions via AURORA API..."):
                        refinement = _with_client(
                            lambda c: c.refine_prompt(
                                message["user_prompt"],
                                intent=message["intent"],
                                profiles=message["bundle"],
                                retrieval=result,
                                answers={},
                                regenerate_on_pivot=False,
                                options=_options(),
                            )
                        )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "kind": "refinement",
                            "intent": message["intent"],
                            "bundle": message["bundle"],
                            "retrieval": result,
                            "user_prompt": message["user_prompt"],
                            "refinement": refinement,
                            "qa_log": [
                                {
                                    "question": q["question"],
                                    "choices": q.get("choices", []),
                                    "answer": None,
                                    "skipped": False,
                                }
                                for q in refinement.get("questions", [])
                            ],
                            "locked": False,
                            "skipped": False,
                            "pivot_resolved": False,
                            "generated": False,
                        }
                    )
                    message["proceeded"] = True
                except AuroraApiError as exc:
                    _append_error("refinement", str(exc), message.get("user_prompt"))
                st.rerun()


def _answers_from_qa(message: dict[str, Any]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for qa in message.get("qa_log", []):
        answer = (qa.get("answer") or "").strip()
        if answer:
            answers[qa["question"]] = answer
    return answers


def _lock_refinement(message: dict[str, Any], *, regenerate_on_pivot: bool) -> None:
    answers = _answers_from_qa(message)
    if not answers and not regenerate_on_pivot:
        message["locked"] = True
        message["skipped"] = True
        message["refinement"] = {
            **message["refinement"],
            "done": True,
            "refined_prompt": message["user_prompt"],
            "questions": [],
            "needs_re_retrieval": False,
        }
        return

    refinement = _with_client(
        lambda c: c.refine_prompt(
            message["user_prompt"],
            intent=message["intent"],
            profiles=message["bundle"],
            retrieval=message["retrieval"],
            answers=answers,
            regenerate_on_pivot=regenerate_on_pivot,
            options=_options(),
        )
    )
    message["refinement"] = refinement
    message["locked"] = True
    message["skipped"] = False
    message["pivot_resolved"] = not refinement.get("needs_re_retrieval")


def _apply_pivot_regenerate(message: dict[str, Any]) -> None:
    _lock_refinement(message, regenerate_on_pivot=True)
    refinement = message["refinement"]
    new_intent = refinement.get("new_intent")
    new_profiles = refinement.get("profiles")
    new_retrieval = refinement.get("retrieval")
    if not (new_intent and new_profiles and new_retrieval):
        message["pivot_resolved"] = True
        message["pivot_decision"] = "keep"
        return

    refined_prompt = refinement.get("refined_prompt") or message["user_prompt"]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "intent",
            "intent": new_intent,
            "user_prompt": refined_prompt,
            "proceeded": True,
        }
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "profiles",
            "intent": new_intent,
            "bundle": new_profiles,
            "user_prompt": refined_prompt,
            "proceeded": True,
        }
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "retrieval",
            "intent": new_intent,
            "bundle": new_profiles,
            "result": new_retrieval,
            "user_prompt": refined_prompt,
            "proceeded": False,
        }
    )
    message["pivot_resolved"] = True
    message["pivot_decision"] = "regenerate"


def _apply_pivot_keep(message: dict[str, Any]) -> None:
    message["pivot_resolved"] = True
    message["pivot_decision"] = "keep"


def _render_refinement_question(message_idx: int, message: dict[str, Any], qa_idx: int) -> None:
    qa = message["qa_log"][qa_idx]
    with st.chat_message("assistant"):
        st.markdown(
            _substep_chip(f"Q{qa_idx + 1} of {len(message['qa_log'])}"),
            unsafe_allow_html=True,
        )
        st.markdown(f"**{qa['question']}**")
        if qa.get("answer"):
            st.success(qa["answer"])
            return
        if qa.get("skipped"):
            st.info("Skipped")
            return
        if message.get("locked") or message.get("skipped"):
            return

        choices = qa.get("choices") or []
        if choices:
            cols = st.columns(min(len(choices), 4))
            for choice_idx, choice in enumerate(choices):
                with cols[choice_idx % len(cols)]:
                    if st.button(
                        choice,
                        key=f"choice_{message_idx}_{qa_idx}_{choice_idx}",
                        use_container_width=True,
                    ):
                        qa["answer"] = choice
                        st.rerun()

        custom = st.text_input(
            "Or type your own answer",
            key=f"custom_{message_idx}_{qa_idx}",
            placeholder="Free text answer",
        )
        submit_col, skip_col = st.columns([2, 1], vertical_alignment="bottom")
        with submit_col:
            if st.button(
                "Submit custom answer",
                key=f"submit_{message_idx}_{qa_idx}",
                disabled=not custom.strip(),
                use_container_width=True,
            ):
                qa["answer"] = custom.strip()
                st.rerun()
        with skip_col:
            if st.button("Skip this question", key=f"skip_q_{message_idx}_{qa_idx}", use_container_width=True):
                qa["skipped"] = True
                st.rerun()


def _render_generate_button(idx: int, message: dict[str, Any]) -> None:
    if message.get("generated"):
        st.caption("Content generated below")
        return
    if st.button("Proceed -> Generate content", key=f"generate_{idx}", type="primary"):
        try:
            _proceed_to_generation(message)
        except AuroraApiError as exc:
            _append_error("generation", str(exc), message.get("user_prompt"))
        st.rerun()


def _render_refinement_footer(idx: int, message: dict[str, Any]) -> None:
    refinement = message["refinement"]
    with st.chat_message("assistant"):
        st.markdown(_substep_chip("Stage 4 · Current refined prompt", kind="footer"), unsafe_allow_html=True)
        refined_prompt = refinement.get("refined_prompt") or message["user_prompt"]
        st.markdown(f"> {refined_prompt.replace(chr(10), chr(10) + '> ')}")
        st.caption(f"Clarifications answered: {len(_answers_from_qa(message))} / {MAX_REFINEMENT_TURNS}")

        if message.get("locked"):
            if not refinement.get("needs_re_retrieval"):
                st.success("Locked in. Existing snippets can be reused.")
                _render_generate_button(idx, message)
                return
            if message.get("pivot_resolved"):
                if message.get("pivot_decision") == "regenerate":
                    st.success("Locked in. Intent pivoted; new profiles and retrieval were appended below.")
                else:
                    st.info("Locked in. Intent pivoted, but current snippets were kept.")
                    _render_generate_button(idx, message)
                return

            new_intent = refinement.get("new_intent") or {}
            st.warning(
                "**Intent pivoted on the refined prompt.** "
                f"New tasks: `{new_intent.get('task_codes')}`; "
                f"sector `{new_intent.get('sector')}`; "
                f"keywords `{new_intent.get('topic_keywords')}`."
            )
            regen_col, keep_col = st.columns(2)
            with regen_col:
                if st.button("Regenerate (new profiles + retrieval)", key=f"regen_{idx}", type="primary", use_container_width=True):
                    try:
                        _apply_pivot_regenerate(message)
                    except AuroraApiError as exc:
                        _append_error("refinement", str(exc), message.get("user_prompt"))
                    st.rerun()
            with keep_col:
                if st.button("Keep current snippets", key=f"keep_{idx}", use_container_width=True):
                    _apply_pivot_keep(message)
                    st.rerun()
            return

        col_lock, col_skip = st.columns(2)
        with col_lock:
            if st.button("Use this prompt", key=f"lock_{idx}", type="primary", use_container_width=True):
                try:
                    _lock_refinement(message, regenerate_on_pivot=False)
                except AuroraApiError as exc:
                    _append_error("refinement", str(exc), message.get("user_prompt"))
                st.rerun()
        with col_skip:
            if st.button("Skip refinement", key=f"skip_refine_{idx}", use_container_width=True):
                message["locked"] = True
                message["skipped"] = True
                message["refinement"] = {
                    **message["refinement"],
                    "done": True,
                    "refined_prompt": message["user_prompt"],
                    "questions": [],
                    "needs_re_retrieval": False,
                }
                st.rerun()


def _render_refinement_message(idx: int, message: dict[str, Any]) -> None:
    with st.chat_message("assistant"):
        st.markdown(
            _block_anchor("refinement", idx) + _stage_chip("refinement"),
            unsafe_allow_html=True,
        )
        st.markdown("**Refining your prompt** — answer each question or skip below.")
    for qa_idx in range(len(message.get("qa_log", []))):
        _render_refinement_question(idx, message, qa_idx)
    _render_refinement_footer(idx, message)


def _active_generation_context(message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    refinement = message["refinement"]
    if message.get("pivot_decision") == "regenerate" and refinement.get("new_intent"):
        return (
            refinement.get("new_intent") or message["intent"],
            refinement.get("profiles") or message["bundle"],
            refinement.get("retrieval") or message["retrieval"],
        )
    return message["intent"], message["bundle"], message["retrieval"]


def _proceed_to_generation(message: dict[str, Any]) -> None:
    intent, profiles, retrieval = _active_generation_context(message)
    refined_prompt = message["refinement"].get("refined_prompt") or message["user_prompt"]
    snippets = retrieval.get("snippets", [])
    with st.spinner("Generating content via AURORA API..."):
        content = _with_client(
            lambda c: c.generate_draft(
                refined_prompt=refined_prompt,
                intent=intent,
                profiles=profiles,
                snippets=snippets,
                options=_options(),
            )
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "content",
            "result": content,
            "intent": intent,
            "bundle": profiles,
            "retrieval": retrieval,
            "refined_prompt": refined_prompt,
            "evaluated": False,
        }
    )
    message["generated"] = True


def _proceed_to_evaluation(message: dict[str, Any]) -> None:
    """Run the KPI evaluation for a generated content message."""
    snippets = message.get("retrieval", {}).get("snippets", [])
    with st.spinner("Evaluating content via AURORA API..."):
        evaluation = _with_client(
            lambda c: c.evaluate_draft(
                refined_prompt=message["refined_prompt"],
                content=message["result"],
                intent=message.get("intent"),
                profiles=message.get("bundle"),
                snippets=snippets,
                options=_options(),
            )
        )
    st.session_state.messages.append(
        {"role": "assistant", "kind": "evaluation", "result": evaluation}
    )
    message["evaluated"] = True


def _render_content_message(idx: int, message: dict[str, Any]) -> None:
    result = message["result"]
    with st.chat_message("assistant"):
        st.markdown(_block_anchor("content", idx) + _stage_chip("content"), unsafe_allow_html=True)
        if result.get("source") == "llm" and result.get("model"):
            st.markdown(f"_Generated via `{result['model']}`._")
        else:
            st.markdown("_Deterministic server stub — configure generation on the server to enable LLM output._")
        body = result.get("body", "")
        if body.strip():
            st.markdown(body)
        else:
            st.warning(
                "No generated content was returned by the server. Generate again, "
                "or check the configured content model."
            )
        citations = result.get("citations", [])
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                snippets = {
                    idx + 1: snippet
                    for idx, snippet in enumerate(message.get("retrieval", {}).get("snippets", []))
                }
                for citation in citations:
                    snippet = snippets.get(citation.get("index"))
                    title = snippet.get("title") if snippet else citation.get("title")
                    score = f" · score={float(snippet.get('score', 0)):.2f}" if snippet else ""
                    st.markdown(
                        f"**[{citation.get('index')}]** {title} — "
                        f"`{citation.get('source_doc')}::{citation.get('node_id')}`{score}"
                    )

        if message.get("evaluated"):
            st.caption("Evaluation completed below")
        else:
            st.caption(
                "The draft has not been evaluated yet — run the KPI generation review "
                f"(channel={st.session_state['channel']}, origin={st.session_state['origin']}, "
                f"strict={'on' if st.session_state['strict_mode'] else 'off'})."
            )
            if st.button(
                "Proceed -> Evaluate draft", key=f"evaluate_{idx}", type="primary"
            ):
                try:
                    _proceed_to_evaluation(message)
                except AuroraApiError as exc:
                    _append_error("evaluation", str(exc), message.get("refined_prompt"))
                st.rerun()


def _verdict_banner(result: dict[str, Any]) -> str:
    if result.get("passed"):
        return (
            '<div style="background:#dcfce7;border-left:4px solid #166534;'
            'padding:10px 14px;border-radius:6px;font-weight:600;color:#14532d;">'
            "Passed — no blocking KPI violations detected.</div>"
        )
    failed = ", ".join(f"`{item}`" for item in result.get("failed_blocking", []))
    return (
        '<div style="background:#fee2e2;border-left:4px solid #b91c1c;'
        'padding:10px 14px;border-radius:6px;font-weight:600;color:#7f1d1d;">'
        f"Blocked — {len(result.get('failed_blocking', []))} blocking KPI failure(s): {failed}</div>"
    )


_WEIGHT_STYLES = {
    "Blocking": ("#fee2e2", "#b91c1c"),
    "High": ("#ffedd5", "#c2410c"),
    "Medium": ("#e2e8f0", "#334155"),
    "Low": ("#f1f5f9", "#64748b"),
}

_MATURITY_STYLES = {
    "high": ("#dcfce7", "#166534"),
    "medium": ("#fef3c7", "#92400e"),
    "low": ("#fee2e2", "#991b1b"),
}

_TIER_LABELS = {
    1: "Tier 1 · Deterministic checks",
    2: "Tier 2 · LLM judges",
    3: "Tier 3 · Human signoff (dCLP)",
}


def _weight_chip(weight: str) -> str:
    bg, fg = _WEIGHT_STYLES.get(weight, _WEIGHT_STYLES["Medium"])
    return (
        f'<span style="background:{bg};color:{fg};padding:1px 8px;'
        f'border-radius:6px;font-weight:600;font-size:0.72em;">{weight}</span>'
    )


def _maturity_chips(maturity: dict[str, str]) -> str:
    chips = []
    for category, level in sorted(maturity.items()):
        bg, fg = _MATURITY_STYLES.get(level, _MATURITY_STYLES["medium"])
        chips.append(
            f'<span style="background:{bg};color:{fg};padding:3px 10px;'
            f'border-radius:8px;font-weight:600;font-size:0.78em;'
            f'display:inline-block;margin:2px 4px 2px 0;">'
            f"{category} · {level}</span>"
        )
    return "".join(chips)


def _kpi_status(kpi: dict[str, Any]) -> tuple[str, str]:
    if kpi.get("tier") == 3:
        return "⏳", "awaiting signoff"
    if kpi.get("source") == "skipped":
        return "⏭️", "skipped"
    if kpi.get("passed"):
        return "✅", "pass"
    return "❌", "fail"


def _kpi_row(kpi: dict[str, Any]) -> str:
    icon, _ = _kpi_status(kpi)
    reason = kpi.get("reason") or ""
    value = kpi.get("value", "")
    return (
        f"{icon} &nbsp;**{kpi.get('name')}** &nbsp;{_weight_chip(kpi.get('weight', 'Medium'))} "
        f"&nbsp;`{value}`  \n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#64748b;font-size:0.85em;'>{reason}</span>"
    )


def _render_evaluation_message(idx: int, message: dict[str, Any]) -> None:
    result = message["result"]
    results = result.get("results", [])
    machine = [k for k in results if k.get("tier") != 3]
    tier3 = [k for k in results if k.get("tier") == 3]
    passed = [k for k in machine if k.get("passed") and k.get("source") != "skipped"]
    failed = [k for k in machine if not k.get("passed")]
    skipped = [k for k in machine if k.get("passed") and k.get("source") == "skipped"]

    with st.chat_message("assistant"):
        st.markdown(
            _block_anchor("evaluation", idx) + _stage_chip("evaluation"),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"_Evaluated via {result.get('source')} "
            f"(channel={result.get('channel')}, origin={result.get('origin')})._"
        )
        st.markdown(_verdict_banner(result), unsafe_allow_html=True)

        # ── At-a-glance counts ─────────────────────────────────────────
        passed_col, failed_col, skipped_col, signoff_col = st.columns(4)
        passed_col.metric("Passed", len(passed))
        failed_col.metric("Failed", len(failed))
        skipped_col.metric("Skipped", len(skipped))
        signoff_col.metric("Signoffs pending", len(tier3))

        # ── Failures first — visible without expanding ─────────────────
        if failed:
            blocking_failed = [k for k in failed if k.get("weight") == "Blocking"]
            st.markdown(
                "**Needs attention** — "
                + (
                    f"{len(blocking_failed)} blocking, {len(failed) - len(blocking_failed)} other"
                    if blocking_failed
                    else f"{len(failed)} non-blocking finding(s)"
                )
            )
            for kpi in sorted(
                failed,
                key=lambda item: (0 if item.get("weight") == "Blocking" else 1, item.get("kpi_id", "")),
            ):
                st.markdown(_kpi_row(kpi), unsafe_allow_html=True)
        else:
            st.markdown("**No machine-check failures.**")

        # ── Maturity rollup ────────────────────────────────────────────
        maturity = result.get("maturity_by_category") or {}
        if maturity:
            st.markdown("**Maturity by category**")
            st.markdown(_maturity_chips(maturity), unsafe_allow_html=True)

        # ── Human signoffs (dCLP) ──────────────────────────────────────
        dclp = result.get("dclp_steps_required") or []
        if dclp:
            st.markdown(
                "**Editorial signoff required (dCLP)** — human steps AURORA "
                "flags but never auto-clears"
            )
            for kpi in tier3:
                st.markdown(_kpi_row(kpi), unsafe_allow_html=True)

        # ── Full breakdown, one expander per tier with pass counts ─────
        for tier in (1, 2, 3):
            tier_results = [k for k in results if k.get("tier") == tier]
            if not tier_results:
                continue
            if tier == 3:
                header = f"{_TIER_LABELS[tier]} — {len(tier_results)} step(s) awaiting signoff"
            else:
                tier_skipped = sum(1 for k in tier_results if k.get("source") == "skipped")
                tier_scored = [k for k in tier_results if k.get("source") != "skipped"]
                tier_passed = sum(1 for k in tier_scored if k.get("passed"))
                header = f"{_TIER_LABELS[tier]} — {tier_passed}/{len(tier_scored)} passed"
                if not tier_scored:
                    header = f"{_TIER_LABELS[tier]} — all {tier_skipped} skipped (no LLM configured)"
                elif tier_skipped:
                    header += f" · {tier_skipped} skipped"
            with st.expander(header):
                for kpi in sorted(
                    tier_results,
                    key=lambda item: (
                        0 if not item.get("passed") else 1,
                        0 if item.get("weight") == "Blocking" else 1,
                        item.get("kpi_id", ""),
                    ),
                ):
                    st.markdown(_kpi_row(kpi), unsafe_allow_html=True)


def _render_error_message(idx: int, message: dict[str, Any]) -> None:
    with st.chat_message("assistant"):
        st.markdown(_block_anchor("error", idx) + _stage_chip("error"), unsafe_allow_html=True)
        st.error(f"{message.get('stage', 'stage')} failed")
        st.code(message.get("message", ""), language="text")


def _timeline_items() -> list[dict[str, Any]]:
    items = []
    for idx, message in enumerate(st.session_state.messages):
        kind = "user" if message.get("role") == "user" else message.get("kind")
        if kind:
            items.append({"idx": idx, "kind": kind, "label": TIMELINE_LABELS.get(kind, kind.title())})
    return items


def _render_pipeline_sidebar() -> None:
    items = _timeline_items()
    if not items:
        st.caption("Pipeline phases will appear here.")
        return
    st.markdown("**Phases**")
    for item in items:
        target = f"block-{item['kind']}-{item['idx']}"
        st.markdown(f"- [{item['label']}](#{target})")


with st.sidebar:
    st.subheader("Server")
    st.session_state["api_base_url"] = st.text_input(
        "API base URL",
        value=st.session_state["api_base_url"],
    ).rstrip("/")
    if st.button("Check server health", use_container_width=True):
        try:
            health = _with_client(lambda c: c.health())
            st.success(f"{health.get('service', 'server')} is {health.get('status', 'ok')}")
        except AuroraApiError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Run options")
    st.session_state["retrieval_backend"] = st.selectbox(
        "Retrieval backend",
        ["pageindex", "vector_rag"],
        index=0 if st.session_state["retrieval_backend"] == "pageindex" else 1,
    )
    st.session_state["retrieval_k"] = st.number_input(
        "Snippets (k)",
        min_value=1,
        max_value=20,
        value=int(st.session_state["retrieval_k"]),
        step=1,
    )
    st.session_state["channel"] = st.selectbox(
        "Channel",
        ["web", "chat", "messages", "employee", "app_ib"],
        index=["web", "chat", "messages", "employee", "app_ib"].index(st.session_state["channel"]),
    )
    st.session_state["origin"] = st.selectbox(
        "Origin",
        ["instant", "human", "genai_knowledge"],
        index=["instant", "human", "genai_knowledge"].index(st.session_state["origin"]),
    )
    st.session_state["strict_mode"] = st.toggle(
        "Strict evaluation mode",
        value=bool(st.session_state["strict_mode"]),
    )
    st.session_state["stage_backgrounds"] = st.toggle(
        "Tint stage bubbles",
        value=bool(st.session_state["stage_backgrounds"]),
    )

    st.divider()
    _render_pipeline_sidebar()
    st.divider()
    if st.button("Clear conversation", disabled=not st.session_state.messages, use_container_width=True):
        st.session_state.messages = []
        st.rerun()


caption = f"Server-backed pipeline · `{st.session_state['api_base_url']}`"
st.caption(caption)

messages = st.session_state.messages


def _latest_idx_of(kind: str) -> int | None:
    return next(
        (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx].get("kind") == kind),
        None,
    )


latest_intent_idx = _latest_idx_of("intent")
latest_profiles_idx = _latest_idx_of("profiles")
latest_retrieval_idx = _latest_idx_of("retrieval")

for idx, message in enumerate(messages):
    if message.get("role") == "user":
        with st.chat_message("user"):
            st.markdown(_block_anchor("user", idx), unsafe_allow_html=True)
            st.markdown(message["content"])
    elif message.get("kind") == "intent":
        _render_intent_message(idx, message, idx == latest_intent_idx)
    elif message.get("kind") == "profiles":
        _render_profiles_message(idx, message, idx == latest_profiles_idx)
    elif message.get("kind") == "retrieval":
        _render_retrieval_message(idx, message, idx == latest_retrieval_idx)
    elif message.get("kind") == "refinement":
        _render_refinement_message(idx, message)
    elif message.get("kind") == "content":
        _render_content_message(idx, message)
    elif message.get("kind") == "evaluation":
        _render_evaluation_message(idx, message)
    elif message.get("kind") == "error":
        _render_error_message(idx, message)


prompt = st.chat_input("Ask AURORA...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.spinner("Classifying intent via AURORA API..."):
            intent = _with_client(lambda c: c.classify_intent(prompt, _options()))
        message = {
            "role": "assistant",
            "kind": "intent",
            "intent": intent,
            "user_prompt": prompt,
            "proceeded": False,
        }
        st.session_state.messages.append(message)
        _render_intent_message(len(st.session_state.messages) - 1, message, True)
    except AuroraApiError as exc:
        _append_error("intent", str(exc), prompt)
        st.rerun()
