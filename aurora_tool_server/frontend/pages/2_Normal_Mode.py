"""Chat-first AURORA pipeline page."""

from __future__ import annotations

import json
from html import escape
from typing import Any

import streamlit as st

from api_client import AuroraApiClient, AuroraApiError
from branding import apply_branding
from normal_mode import (
    NORMAL_LATEST_RUN_KEY,
    NORMAL_MESSAGES_KEY,
    NORMAL_PENDING_KEY,
    NORMAL_SHOW_DETAILS_KEY,
    answer_summary,
    answers_by_question,
    assistant_message_from_run,
    build_pending_from_run,
    clarification_message,
    compact_evaluation_verdict,
    continue_after_clarification,
    init_normal_mode_state,
    pending_questions,
    pipeline_options,
)
from settings_state import init_pipeline_state


TASK_LABELS = {
    "T1_DRAFT": "Draft new content",
    "T1_TRANSLATE": "Translate existing content",
    "T1_SEARCH": "Search corpus for related articles",
    "T2_COMPLIANCE": "Quality and compliance check",
    "T4_RENEWAL": "Detect and renew aging articles",
}


st.set_page_config(page_title="Normal mode · AURORA", layout="centered")
apply_branding("Normal mode", "Chat-first AURORA pipeline")

init_pipeline_state()
init_normal_mode_state(st.session_state)

with st.sidebar:
    if st.button("Clear chat", key="normal_clear_chat", use_container_width=True):
        st.session_state[NORMAL_MESSAGES_KEY] = []
        st.session_state[NORMAL_PENDING_KEY] = None
        st.session_state[NORMAL_LATEST_RUN_KEY] = None
        st.rerun()


def _options() -> dict[str, Any]:
    return pipeline_options(st.session_state)


def _with_client(call):
    client = AuroraApiClient(st.session_state["api_base_url"])
    try:
        return call(client)
    finally:
        client.close()


def _append_assistant(content: str, *, kind: str = "assistant") -> None:
    st.session_state[NORMAL_MESSAGES_KEY].append(
        {"role": "assistant", "content": content, "kind": kind}
    )


def _append_error(message: str) -> None:
    _append_assistant(f"AURORA API error: {message}", kind="error")


def _run_initial_prompt(prompt: str) -> None:
    run = _with_client(
        lambda client: client.run_pipeline(
            prompt,
            refinement_policy="ask_first",
            options=_options(),
        )
    )
    st.session_state[NORMAL_LATEST_RUN_KEY] = run

    if run.get("status") == "needs_clarification":
        pending = build_pending_from_run(prompt, run)
        st.session_state[NORMAL_PENDING_KEY] = pending
        _append_assistant(clarification_message(pending), kind="clarification")
        return

    st.session_state[NORMAL_PENDING_KEY] = None
    _append_assistant(assistant_message_from_run(run), kind="final")


def _continue_pending_run(pending: dict[str, Any], answers: dict[str, str]) -> None:
    run = _with_client(
        lambda client: continue_after_clarification(
            client,
            pending,
            answers,
            _options(),
        )
    )
    st.session_state[NORMAL_PENDING_KEY] = None
    st.session_state[NORMAL_LATEST_RUN_KEY] = run
    st.session_state[NORMAL_MESSAGES_KEY].append(
        {"role": "user", "content": answer_summary(answers), "kind": "answers"}
    )
    _append_assistant(assistant_message_from_run(run), kind="final")


def _status_counts(evaluation: dict[str, Any]) -> tuple[int, int, int]:
    results = evaluation.get("results") or []
    machine = [item for item in results if item.get("tier") != 3]
    passed = sum(1 for item in machine if item.get("passed") and item.get("source") != "skipped")
    failed = sum(1 for item in machine if not item.get("passed"))
    signoff = len(evaluation.get("dclp_steps_required") or [])
    return passed, failed, signoff


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message.get("kind") == "error":
            st.error(message["content"])
        else:
            st.markdown(message["content"])


def _render_clarification_form(pending: dict[str, Any]) -> None:
    questions = pending_questions(pending)
    if not questions:
        return

    run_id = pending.get("run_id") or "pending"
    with st.chat_message("assistant"):
        with st.form(f"normal_clarification_{run_id}"):
            collected: list[str] = []
            for index, question in enumerate(questions):
                label = question.get("question", f"Question {index + 1}")
                choices = question.get("choices") or []
                key_base = f"normal_{run_id}_{index}"
                if choices:
                    selected = st.radio(label, choices, key=f"{key_base}_choice")
                    custom = st.text_input(
                        f"Custom answer {index + 1}",
                        key=f"{key_base}_custom",
                        label_visibility="collapsed",
                        placeholder="Custom answer",
                    )
                    collected.append(custom.strip() or selected)
                else:
                    collected.append(st.text_area(label, key=f"{key_base}_text", height=88))

            submitted = st.form_submit_button("Continue", type="primary")

    if not submitted:
        return

    answers = answers_by_question(pending, collected)
    if len(answers) < len(questions):
        st.warning("Please answer every question before continuing.")
        return

    try:
        with st.spinner("Running AURORA..."):
            _continue_pending_run(pending, answers)
    except AuroraApiError as exc:
        _append_error(str(exc))
    st.rerun()


def _html_text(value: Any) -> str:
    return escape(str(value if value is not None else "-"))


def _html_pre(value: Any) -> str:
    return escape(json.dumps(value or {}, indent=2, ensure_ascii=False))


def _drawer_section(title: str, body: str, *, open_by_default: bool = False) -> str:
    opened = " open" if open_by_default else ""
    return (
        f"<details class='normal-drawer-section'{opened}>"
        f"<summary>{_html_text(title)}</summary>"
        f"<div class='normal-drawer-section-body'>{body}</div>"
        "</details>"
    )


def _intent_html(intent: dict[str, Any] | None) -> str:
    if not intent:
        return ""
    tasks = ", ".join(
        f"{code} ({TASK_LABELS.get(code, code)})" for code in intent.get("task_codes", [])
    )
    items = [
        ("Role", intent.get("role")),
        ("Tasks", tasks or "-"),
        ("Confidence", f"{float(intent.get('confidence', 0)):.2f}"),
        ("Sector", intent.get("sector")),
        ("Keywords", ", ".join(intent.get("topic_keywords") or [])),
        ("Reason", intent.get("task_reason")),
    ]
    body = "".join(
        f"<p><strong>{_html_text(label)}</strong><br>{_html_text(value)}</p>"
        for label, value in items
        if value
    )
    return _drawer_section("Intent", body, open_by_default=True)


def _profiles_html(profiles: dict[str, Any] | None) -> str:
    if not profiles:
        return ""
    parts = []
    for label, items in [
        ("Workflow", profiles.get("workflow") or []),
        ("Domain experts", profiles.get("domain_expert") or []),
    ]:
        rows = "".join(
            "<li>"
            f"<strong>{_html_text(profile.get('name'))}</strong> "
            f"<code>{_html_text(profile.get('id'))}</code>"
            f"<br><span>{_html_text(profile.get('selection_reason') or '')}</span>"
            "</li>"
            for profile in items
        )
        parts.append(f"<h4>{_html_text(label)} ({len(items)})</h4><ul>{rows or '<li>None</li>'}</ul>")
    return _drawer_section("Profiles", "".join(parts))


def _retrieval_html(retrieval: dict[str, Any] | None) -> str:
    if not retrieval:
        return ""
    snippets = retrieval.get("snippets") or []
    summary = (
        f"<p><strong>{_html_text(retrieval.get('provider'))}</strong> · "
        f"{len(snippets)} snippet(s) · {_html_text(retrieval.get('source', 'deterministic'))}</p>"
    )
    snippet_rows = []
    for index, snippet in enumerate(snippets, 1):
        body = str(snippet.get("content") or "")
        if len(body) > 900:
            body = body[:900].rstrip() + "\n\n(truncated)"
        snippet_rows.append(
            "<details class='normal-nested-detail'>"
            f"<summary>{index}. {_html_text(snippet.get('title') or snippet.get('source_doc'))}</summary>"
            f"<p><code>{_html_text(snippet.get('source_doc'))}::{_html_text(snippet.get('node_id'))}</code> · "
            f"score={float(snippet.get('score', 0)):.2f}</p>"
            f"<p>{_html_text(snippet.get('reason') or '')}</p>"
            f"<pre>{_html_text(body)}</pre>"
            "</details>"
        )
    return _drawer_section("Retrieval", summary + "".join(snippet_rows))


def _refinement_html(refinement: dict[str, Any] | None) -> str:
    if not refinement:
        return ""
    questions = refinement.get("questions") or []
    body = (
        f"<p>Done: {_html_text(bool(refinement.get('done')))} · "
        f"Questions: {len(questions)} · "
        f"Re-retrieval: {_html_text(bool(refinement.get('needs_re_retrieval')))}</p>"
        f"<pre>{_html_text(refinement.get('refined_prompt') or '')}</pre>"
    )
    return _drawer_section("Refinement", body)


def _content_html(content: dict[str, Any] | None) -> str:
    if not content:
        return ""
    body = str(content.get("body") or "")
    summary = (
        f"<p>{len(body)} characters · {len(content.get('citations') or [])} citation(s) · "
        f"{_html_text(content.get('source', 'deterministic'))}</p>"
    )
    return _drawer_section("Content", summary + f"<pre>{_html_text(body)}</pre>")


def _evaluation_html(evaluation: dict[str, Any] | None) -> str:
    if not evaluation:
        return ""
    passed, failed, signoff = _status_counts(evaluation)
    verdict = compact_evaluation_verdict(evaluation).replace("**", "").replace("`", "")
    rows = "".join(
        "<li>"
        f"<strong>{_html_text(item.get('name'))}</strong> "
        f"<code>{_html_text(item.get('kpi_id'))}</code> · "
        f"{'pass' if item.get('passed') else 'fail'} · {_html_text(item.get('weight', 'Medium'))}"
        f"<br><span>{_html_text(item.get('reason') or '')}</span>"
        "</li>"
        for item in evaluation.get("results") or []
    )
    body = (
        f"<p>{_html_text(verdict)}</p>"
        "<div class='normal-drawer-metrics'>"
        f"<span><strong>{passed}</strong><br>Passed</span>"
        f"<span><strong>{failed}</strong><br>Failed</span>"
        f"<span><strong>{signoff}</strong><br>Signoff</span>"
        "</div>"
        f"<ul>{rows or '<li>No KPI results returned.</li>'}</ul>"
    )
    return _drawer_section("Evaluation", body)


def _audit_html(audit: dict[str, Any] | None) -> str:
    if not audit:
        return ""
    events = audit.get("events") or []
    rows = []
    for event in events:
        if event.get("error"):
            detail = f"<pre>{_html_text(event['error'])}</pre>"
        else:
            detail = (
                "<pre>"
                f"{_html_pre({'input_summary': event.get('input_summary', {}), 'output_summary': event.get('output_summary', {})})}"
                "</pre>"
            )
        rows.append(
            "<details class='normal-nested-detail'>"
            f"<summary>{_html_text(event.get('stage'))} · {_html_text(event.get('source', 'system'))}</summary>"
            f"{detail}"
            "</details>"
        )
    body = f"<p><code>{_html_text(audit.get('run_id'))}</code> · {len(events)} event(s)</p>" + "".join(rows)
    return _drawer_section("Audit", body)


def _detail_drawer_html(payload: dict[str, Any] | None) -> str:
    if payload:
        body = "".join(
            [
                _intent_html(payload.get("intent")),
                _profiles_html(payload.get("profiles")),
                _retrieval_html(payload.get("retrieval")),
                _refinement_html(payload.get("refinement")),
                _content_html(payload.get("content")),
                _evaluation_html(payload.get("evaluation")),
                _audit_html(payload.get("audit")),
            ]
        )
        subtitle = f"Run {_html_text(payload.get('run_id', '-'))} · {_html_text(payload.get('status', 'pending'))}"
    else:
        body = "<p class='normal-empty-detail'>Run details will appear after the first message.</p>"
        subtitle = "No active run"

    return f"""
    <style>
      .normal-detail-drawer {{
        position: fixed;
        top: 4.75rem;
        right: 1.25rem;
        width: min(430px, calc(100vw - 2rem));
        max-height: calc(100vh - 6rem);
        overflow-y: auto;
        z-index: 99991;
        background: #fbfcfe;
        border-left: 1px solid #e5e7eb;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.22);
        padding: 0;
        color: #111827;
      }}
      .normal-drawer-header {{
        position: sticky;
        top: 0;
        padding: 0.85rem 1rem;
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        z-index: 1;
      }}
      .normal-drawer-title {{
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.25;
      }}
      .normal-drawer-subtitle {{
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.12rem;
      }}
      .normal-drawer-content {{
        padding: 0.7rem 0.8rem 0.9rem;
      }}
      .normal-drawer-section {{
        border: 1px solid #dde4ee;
        border-radius: 8px;
        margin: 0 0 0.6rem;
        background: #ffffff;
        overflow: hidden;
      }}
      .normal-drawer-section summary {{
        cursor: pointer;
        font-weight: 650;
        padding: 0.68rem 0.78rem;
        color: #182235;
        list-style-position: outside;
      }}
      .normal-drawer-section-body {{
        border-top: 1px solid #f3f4f6;
        padding: 0.2rem 0.78rem 0.78rem;
      }}
      .normal-drawer-section p,
      .normal-drawer-section li {{
        color: #334155;
        font-size: 0.82rem;
        line-height: 1.45;
      }}
      .normal-drawer-section h4 {{
        font-size: 0.86rem;
        margin: 0.65rem 0 0.2rem;
        color: #111827;
      }}
      .normal-drawer-section ul {{
        padding-left: 1.05rem;
      }}
      .normal-drawer-section pre {{
        white-space: pre-wrap;
        word-break: break-word;
        background: #f8fafc;
        border: 1px solid #eef2f7;
        border-radius: 6px;
        padding: 0.55rem;
        font-size: 0.76rem;
        line-height: 1.38;
        color: #1f2937;
      }}
      .normal-nested-detail {{
        margin: 0.45rem 0;
      }}
      .normal-drawer-metrics {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.4rem;
        margin: 0.7rem 0;
      }}
      .normal-drawer-metrics span {{
        border: 1px solid #e5e7eb;
        border-radius: 7px;
        padding: 0.45rem;
        text-align: center;
        font-size: 0.78rem;
        background: #f8fafc;
      }}
      .normal-empty-detail {{
        color: #6b7280;
        font-size: 0.9rem;
        margin: 0.25rem 0.2rem 0.4rem;
      }}
      @media (max-width: 720px) {{
        .normal-detail-drawer {{
          top: 4.25rem;
          right: 0.65rem;
          left: 0.65rem;
          width: auto;
          max-height: calc(100vh - 5rem);
        }}
      }}
    </style>
    <aside class="normal-detail-drawer" role="dialog" aria-label="Run details">
      <div class="normal-drawer-header">
        <div class="normal-drawer-title">Details</div>
        <div class="normal-drawer-subtitle">{subtitle}</div>
      </div>
      <div class="normal-drawer-content">
        {body}
      </div>
    </aside>
    """


def _details_payload() -> dict[str, Any] | None:
    return st.session_state.get(NORMAL_LATEST_RUN_KEY) or st.session_state.get(NORMAL_PENDING_KEY)


def _render_details_panel() -> None:
    st.markdown(_detail_drawer_html(_details_payload()), unsafe_allow_html=True)


toggle_label = "Hide details" if st.session_state[NORMAL_SHOW_DETAILS_KEY] else "Show details"
if st.button(toggle_label):
    st.session_state[NORMAL_SHOW_DETAILS_KEY] = not st.session_state[NORMAL_SHOW_DETAILS_KEY]
    st.rerun()

for message in st.session_state[NORMAL_MESSAGES_KEY]:
    _render_message(message)

pending = st.session_state.get(NORMAL_PENDING_KEY)
if pending:
    _render_clarification_form(pending)

if st.session_state[NORMAL_SHOW_DETAILS_KEY]:
    _render_details_panel()

prompt = st.chat_input(
    "Ask AURORA to draft, search, translate, renew, or check content",
    disabled=st.session_state.get(NORMAL_PENDING_KEY) is not None,
)

if prompt:
    st.session_state[NORMAL_MESSAGES_KEY].append(
        {"role": "user", "content": prompt, "kind": "prompt"}
    )
    try:
        with st.spinner("Running AURORA..."):
            _run_initial_prompt(prompt)
    except AuroraApiError as exc:
        _append_error(str(exc))
    st.rerun()
