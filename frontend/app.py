"""AURORA — Streamlit entry point.

Five-stage pipeline driven by the chat:
1. User prompt → intent classification (backend/intent).
2. Proceed → profile selection (backend/profile_selection).
3. Proceed → context retrieval (backend/retrieval, PageIndex provider).
4. Proceed → prompt refinement (backend/prompt_refinement).
5. Proceed → content generation (backend/content_generation).

API key + model are configured on the Settings page; intent / retrieval /
content all fall back to deterministic paths when those are unset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.content_generation import (  # noqa: E402
    ContentRequest,
    ContentResult,
    generate_content,
)
from backend.evaluation import (  # noqa: E402
    EvaluationResult,
    KPIResult,
    evaluate,
)
from backend.intent import IntentResult, classify_full  # noqa: E402
from backend.profile_selection import select  # noqa: E402
from backend.prompt_refinement import (  # noqa: E402
    QuestionWithChoices,
    RefinedPrompt,
    RefinementTurn,
    advance_turn,
    append_qa,
    needs_re_retrieval,
)
from backend.prompt_refinement.service import overwrite_prompt  # noqa: E402
from backend.retrieval import (  # noqa: E402
    PageIndexProvider,
    RetrievalQuery,
    RetrievalResult,
    build_query,
    retrieve,
)
from backend.retrieval.corpus_loader import load_corpora  # noqa: E402
from config import TASK_LABELS  # noqa: E402
from profiles import ProfileBundle  # noqa: E402

MAX_REFINEMENT_TURNS = 5

# Per-stage avatar + chip styling. Keeps each pipeline step visually distinct
# so a fast scroll over the chat history reads as colour-coded stages rather
# than a wall of identical assistant bubbles.
_STAGE_STYLES: dict[str, tuple[str, str, str, str]] = {
    # kind          (avatar, chip_bg,   chip_fg,   chip_label)
    "intent":     ("🧭", "#dbeafe", "#1e40af", "Stage 1 · Intent"),
    "profiles":   ("👥", "#dcfce7", "#166534", "Stage 2 · Profiles"),
    "retrieval":  ("📚", "#ede9fe", "#5b21b6", "Stage 3 · Retrieval"),
    "refinement": ("✏️", "#fef3c7", "#92400e", "Stage 4 · Refinement"),
    "content":    ("📝", "#e2e8f0", "#0f172a", "Stage 5 · Content"),
    "evaluation": ("🛡️", "#cffafe", "#155e75", "Stage 6 · Evaluation"),
}


def _stage_chip(kind: str) -> str:
    """Inline HTML chip rendered at the top of each stage's lead bubble.

    The ``data-stage-chip`` attribute lets the injected CSS tint the parent
    bubble based on which stage it belongs to.
    """
    _, bg, fg, label = _STAGE_STYLES[kind]
    return (
        f'<span data-stage-chip="{kind}" '
        f'style="background:{bg};color:{fg};padding:3px 12px;'
        f'border-radius:10px;font-weight:600;font-size:0.82em;'
        f'letter-spacing:0.02em;">{label}</span>'
    )


def _block_anchor(kind: str, idx: int) -> str:
    """Invisible anchor target used by the sidebar timeline.

    ID format ``block-{kind}-{idx}`` is shared verbatim with the timeline
    item ``data-timeline-target`` so the scroll-spy script can map between
    the two without parsing. ``scroll-margin-top`` keeps the bubble below
    Streamlit's header bar after a programmatic scroll.
    """
    return (
        f'<span id="block-{kind}-{idx}" data-block-anchor="block-{kind}-{idx}" '
        f'style="display:block;height:0;scroll-margin-top:5rem;"></span>'
    )


def _substep_chip(
    label: str,
    bg: str = "#fff7ed",
    fg: str = "#9a3412",
    kind: str = "question",
) -> str:
    """Smaller chip used to number sub-steps inside one stage (refinement Qs).

    ``kind`` ends up as ``data-substep-chip`` so the CSS can colour question
    bubbles and the refined-prompt footer differently.
    """
    return (
        f'<span data-substep-chip="{kind}" '
        f'style="background:{bg};color:{fg};padding:1px 8px;'
        f'border-radius:6px;font-weight:500;font-size:0.72em;">{label}</span>'
    )

st.set_page_config(page_title="AURORA", layout="centered")

st.title("AURORA")

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
st.session_state.setdefault("messages", [])
st.session_state.setdefault("stage_backgrounds", True)


def _inject_stage_styles() -> None:
    """Tint each chat-message bubble by the stage chip it carries.

    Uses ``:has()`` to select the bubble that contains a chip with a
    matching ``data-stage-chip`` attribute. Modern browsers support this.
    Toggled by the Settings-page `Tint stage bubbles` switch.
    """
    if not st.session_state.get("stage_backgrounds", True):
        return
    st.markdown(
        """
        <style>
          /* Shared spacing for every tinted stage / substep bubble. The
             per-rule blocks below only set colour + border so the inner
             padding stays consistent and content never touches the tint
             boundary on either side. */
          [data-testid="stChatMessage"]:is(
            :has([data-stage-chip="intent"]),
            :has([data-stage-chip="profiles"]),
            :has([data-stage-chip="retrieval"]),
            :has([data-stage-chip="refinement"]),
            :has([data-stage-chip="content"]),
            :has([data-stage-chip="evaluation"]),
            :has([data-substep-chip="question"]),
            :has([data-substep-chip="footer"])
          ) {
            padding: 0.9rem 1.25rem 0.9rem 1rem;
            margin-right: 0.75rem;
            box-sizing: border-box;
          }
          /* Anything inside the bubble (markdown, code, columns, expanders)
             should respect the same right gutter so wide blocks don't
             stretch flush to the tint edge. */
          [data-testid="stChatMessage"]:is(
            :has([data-stage-chip="intent"]),
            :has([data-stage-chip="profiles"]),
            :has([data-stage-chip="retrieval"]),
            :has([data-stage-chip="refinement"]),
            :has([data-stage-chip="content"]),
            :has([data-stage-chip="evaluation"]),
            :has([data-substep-chip="question"]),
            :has([data-substep-chip="footer"])
          ) > div {
            min-width: 0;
            max-width: 100%;
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
          [data-testid="stChatMessage"]:has([data-substep-chip="question"]) {
            background: #fff7ed55;
            border-left: 3px solid #c2410c;
            border-radius: 8px;
            /* Indent the substep so it reads as nested under the stage,
               but cap the width so the right edge still lines up with
               the stage bubbles (margin-right 0.75rem is set in the
               shared rule above). */
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

key_set = bool(st.session_state["intent_api_key"])
model_set = bool(st.session_state["intent_model"])
if key_set and model_set:
    st.caption(f"Pipeline: LLM · `{st.session_state['intent_model']}`")
else:
    st.caption("Pipeline: deterministic fallbacks — set the Intent API Key on **Settings**.")


# ── Retrieval provider (corpora cached once per process) ───────────────────


@st.cache_resource
def _cached_corpora():
    return load_corpora()


def _make_retrieval_provider() -> PageIndexProvider:
    # Retrieval (PageIndex) uses its own env-scoped key, independent of the
    # Settings-page key (which only drives intent classification). Model still
    # comes from Settings since intent and retrieval share that today.
    return PageIndexProvider(
        api_key=os.getenv("OPENAI_API_KEY_PAGEINDEX"),
        model=st.session_state["intent_model"] or None,
        corpora=_cached_corpora(),
    )


# ── Markdown renderers ─────────────────────────────────────────────────────


def _intent_markdown(r: IntentResult, source: str) -> str:
    task_cells = ", ".join(f"`{c}` — {TASK_LABELS.get(c, c)}" for c in r.task_codes)
    rows = [
        f"| Role | {r.role} |",
        f"| Tasks | {task_cells} |",
        f"| Confidence | {r.confidence:.2f} |",
        f"| Reason | {r.task_reason} |",
    ]
    if r.sector:
        rows.append(f"| Sector | {r.sector} |")
    if r.topic_keywords:
        kw = ", ".join(f"`{k}`" for k in r.topic_keywords)
        rows.append(f"| Topic keywords | {kw} |")
    if r.language:
        rows.append(f"| Language | {r.language} |")
    return (
        f"**Intent classification** _(via {source})_\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )


def _profiles_markdown(intent: IntentResult, bundle: ProfileBundle) -> str:
    parts = ["**Filtered profiles** _(based on intent above)_\n"]
    if bundle.is_empty():
        parts.append(
            f"_No profiles matched._ Intent tasks: `{intent.task_codes}`, "
            f"sector: `{intent.sector}`."
        )
        return "\n".join(parts)

    parts.append(f"**Workflow** ({len(bundle.workflow)})")
    if not bundle.workflow:
        parts.append("- _none matched_")
    else:
        intent_set = set(intent.task_codes)
        for w in bundle.workflow:
            activated_by = [c for c in w.activates_on_intent_codes if c in intent_set]
            tags = ", ".join(f"`{c}`" for c in activated_by) or "—"
            parts.append(f"- 🛠 **{w.name}** (`{w.id}`) — activated by {tags}")

    parts.append(f"\n**Domain experts** ({len(bundle.domain_expert)})")
    if not bundle.domain_expert:
        parts.append("- _none matched_")
    else:
        kw_lower = {k.lower() for k in intent.topic_keywords}
        for e in bundle.domain_expert:
            matched = [k for k in e.topic_keywords if k.lower() in kw_lower]
            if matched:
                kw_str = ", ".join(f"`{k}`" for k in matched)
            elif kw_lower:
                kw_str = "_no keyword overlap (matched on sector only)_"
            else:
                kw_str = "_no keyword filter applied_"
            parts.append(
                f"- 🎯 **{e.name}** (`{e.id}`)  \n"
                f"    Sector: {e.sector}  \n"
                f"    Matched keywords: {kw_str}"
            )
    return "\n".join(parts)


def _retrieval_markdown(query: RetrievalQuery, result: RetrievalResult) -> str:
    corpora = ", ".join(f"`{c}`" for c in result.corpora_searched) or "—"
    return (
        f"**Context retrieval** _(via {result.provider} · {result.source})_  \n"
        f"Corpora searched: {corpora}  \n"
        f"Query: k={query.k}, task_codes=`{query.task_codes}`, sector=`{query.sector}`  \n"
        f"Snippets returned: **{len(result.snippets)}**\n"
    )


# ── Component renderers ────────────────────────────────────────────────────


def _render_workflow_expander(w):
    with st.expander(f"🛠 {w.name} (`{w.id}`) — details"):
        st.markdown(f"**Description:** {w.description}")
        if w.activates_on_intent_codes:
            st.markdown(
                "**Activates on:** "
                + ", ".join(f"`{c}`" for c in w.activates_on_intent_codes)
            )
        if w.skills:
            st.markdown("**Skills:**")
            for s in w.skills:
                st.markdown(f"- {s}")
        if w.knowledge:
            st.markdown("**Knowledge:**")
            for k in w.knowledge:
                st.markdown(f"- {k}")


def _render_expert_expander(e):
    with st.expander(f"🎯 {e.name} (`{e.id}`) — details"):
        st.markdown(f"**Description:** {e.description}")
        st.markdown(f"**Sector:** {e.sector}")
        if e.topic_keywords:
            st.markdown(
                "**Topic keywords:** " + ", ".join(f"`{k}`" for k in e.topic_keywords)
            )
        if e.expertise_areas:
            st.markdown("**Expertise areas:**")
            for a in e.expertise_areas:
                st.markdown(f"- {a}")


def _render_snippet_expander(i: int, snippet) -> None:
    label = (
        f"#{i + 1} · {snippet.title} · `{snippet.source_doc}` · "
        f"score={snippet.score:.2f}"
    )
    with st.expander(label):
        st.markdown(f"**Reason:** {snippet.reason}")
        if snippet.line_num is not None:
            st.markdown(
                f"**Location:** node `{snippet.node_id}`, line/page {snippet.line_num}"
            )
        body = snippet.content
        if len(body) > 2000:
            body = body[:2000].rstrip() + "\n\n…_(truncated)_"
        st.markdown("**Content:**")
        st.markdown(body)


def _render_intent_message(idx: int, m: dict, is_latest_intent: bool) -> None:
    with st.chat_message("assistant", avatar=_STAGE_STYLES["intent"][0]):
        st.markdown(
            _block_anchor("intent", idx) + _stage_chip("intent"),
            unsafe_allow_html=True,
        )
        st.markdown(_intent_markdown(m["intent"], m["source"]))
        if m.get("raw"):
            with st.expander("Raw LLM output"):
                st.code(m["raw"], language="json")
        if is_latest_intent and not m["proceeded"]:
            if st.button("Proceed → Filter profiles", key=f"proceed_{idx}", type="primary"):
                with st.spinner("Filtering profiles…"):
                    bundle = select(m["intent"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "kind": "profiles",
                        "intent": m["intent"],
                        "bundle": bundle,
                        "user_prompt": m["user_prompt"],
                        "proceeded": False,
                    }
                )
                m["proceeded"] = True
                st.rerun()


def _render_profiles_message(idx: int, m: dict, is_latest_profiles: bool) -> None:
    intent: IntentResult = m["intent"]
    bundle: ProfileBundle = m["bundle"]
    with st.chat_message("assistant", avatar=_STAGE_STYLES["profiles"][0]):
        st.markdown(
            _block_anchor("profiles", idx) + _stage_chip("profiles"),
            unsafe_allow_html=True,
        )
        st.markdown(_profiles_markdown(intent, bundle))
        for w in bundle.workflow:
            _render_workflow_expander(w)
        for e in bundle.domain_expert:
            _render_expert_expander(e)
        if is_latest_profiles and not m.get("proceeded"):
            k_col, btn_col = st.columns([1, 3], vertical_alignment="bottom")
            with k_col:
                k_value = st.number_input(
                    "Snippets (k)",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("retrieval_k", 5),
                    step=1,
                    key=f"k_{idx}",
                    help="How many ranked snippets the retrieval step should return.",
                )
            with btn_col:
                proceed = st.button(
                    "Proceed → Retrieve context",
                    key=f"retrieve_{idx}",
                    type="primary",
                )
            if proceed:
                st.session_state["retrieval_k"] = int(k_value)
                spinner_label = (
                    "Retrieving context via LLM ranker…"
                    if os.getenv("OPENAI_API_KEY_PAGEINDEX") and st.session_state["intent_model"]
                    else "Retrieving context (deterministic keyword match)…"
                )
                with st.spinner(spinner_label):
                    provider = _make_retrieval_provider()
                    query = build_query(m["user_prompt"], intent, bundle, k=int(k_value))
                    result = retrieve(query, providers=[provider])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "kind": "retrieval",
                        "query": query,
                        "result": result,
                        "intent": intent,
                        "bundle": bundle,
                        "user_prompt": m["user_prompt"],
                        "proceeded": False,
                    }
                )
                m["proceeded"] = True
                st.rerun()


def _render_retrieval_message(idx: int, m: dict, is_latest_retrieval: bool) -> None:
    query: RetrievalQuery = m["query"]
    result: RetrievalResult = m["result"]
    with st.chat_message("assistant", avatar=_STAGE_STYLES["retrieval"][0]):
        st.markdown(
            _block_anchor("retrieval", idx) + _stage_chip("retrieval"),
            unsafe_allow_html=True,
        )
        st.markdown(_retrieval_markdown(query, result))
        if not result.snippets:
            st.markdown("_No snippets found._")
            return
        for i, s in enumerate(result.snippets):
            _render_snippet_expander(i, s)

        if is_latest_retrieval and not m.get("proceeded"):
            if st.button(
                "Proceed → Refine prompt",
                key=f"refine_proceed_{idx}",
                type="primary",
            ):
                with st.spinner("Preparing the first refinement question…"):
                    _seed_refinement(
                        user_prompt=m["user_prompt"],
                        intent=m["intent"],
                        bundle=m["bundle"],
                        result=result,
                    )
                m["proceeded"] = True
                st.rerun()


# ── Refinement helpers ─────────────────────────────────────────────────────


def _seed_refinement(
    *,
    user_prompt: str,
    intent: IntentResult,
    bundle: ProfileBundle,
    result: RetrievalResult,
) -> None:
    """Append the initial refinement message + first batch of questions."""
    refined = RefinedPrompt(original=user_prompt, refined=user_prompt)
    _, output = advance_turn(
        original_prompt=user_prompt,
        refined_prompt=user_prompt,
        intent=intent,
        profiles=bundle,
        retrieval=result,
        prior_turns=[],
        api_key=st.session_state["intent_api_key"] or None,
        model=st.session_state["intent_model"] or None,
    )
    qa_log = [
        {"question": q.question, "choices": list(q.choices), "answer": None}
        for q in output.questions
    ]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "refinement",
            "user_prompt": user_prompt,
            "intent": intent,
            "bundle": bundle,
            "retrieval": result,
            "refined": refined,
            "qa_log": qa_log,
            "pending_idx": 0,
            "last_proposed": output.proposed_prompt,
            "generator_done": output.done,
            "locked": False,
            "skipped": False,
        }
    )


def _retrieve_with_intent(
    user_prompt: str, intent: IntentResult, bundle: ProfileBundle
) -> tuple[RetrievalQuery, RetrievalResult]:
    provider = _make_retrieval_provider()
    k = int(st.session_state.get("retrieval_k", 5))
    query = build_query(user_prompt, intent, bundle, k=k)
    result = retrieve(query, providers=[provider])
    return query, result


def _commit_refinement(m: dict) -> None:
    """Lock-in: re-classify intent and *flag* whether a pivot happened.

    No auto-rerun anymore — when ``pivot=True`` the user picks Regenerate /
    Keep current snippets via buttons in the footer (see
    :func:`_apply_pivot_regenerate`).
    """
    refined: RefinedPrompt = m["refined"]
    original_intent: IntentResult = m["intent"]
    refined_text = refined.refined.strip() or refined.original

    with st.spinner("Re-classifying intent on refined prompt…"):
        new_intent, src = classify_full(
            refined_text,
            api_key=st.session_state["intent_api_key"] or None,
            model=st.session_state["intent_model"] or None,
        )
    pivot = needs_re_retrieval(original_intent, new_intent)
    m["new_intent"] = new_intent
    m["intent_source"] = src
    m["pivot"] = pivot
    m["pivot_resolved"] = not pivot  # no pivot ⇒ nothing left to resolve
    m["refined_text"] = refined_text
    m["refined"] = refined.model_copy(update={"locked_in": True})
    m["locked"] = True


def _apply_pivot_regenerate(m: dict) -> None:
    """User chose Regenerate — re-run profiles + retrieval on the new intent."""
    new_intent: IntentResult = m["new_intent"]
    refined_text: str = m["refined_text"]
    src: str = m.get("intent_source", "deterministic")

    with st.spinner("Re-running profiles + retrieval with the pivoted intent…"):
        new_bundle = select(new_intent)
        query, result = _retrieve_with_intent(refined_text, new_intent, new_bundle)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "intent",
            "intent": new_intent,
            "source": src,
            "raw": new_intent.model_dump_json(indent=2) if src == "llm" else None,
            "user_prompt": refined_text,
            "proceeded": True,
        }
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "profiles",
            "intent": new_intent,
            "bundle": new_bundle,
            "user_prompt": refined_text,
            "proceeded": True,
        }
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "retrieval",
            "query": query,
            "result": result,
            "intent": new_intent,
            "bundle": new_bundle,
            "user_prompt": refined_text,
            "proceeded": False,
        }
    )
    m["pivot_resolved"] = True
    m["pivot_decision"] = "regenerate"


def _apply_pivot_keep(m: dict) -> None:
    """User chose to keep the existing snippets despite intent pivot."""
    m["pivot_resolved"] = True
    m["pivot_decision"] = "keep"


def _refinement_prior_turns(m: dict) -> list[RefinementTurn]:
    """Flatten qa_log into the RefinementTurn list the generator expects."""
    turns: list[RefinementTurn] = []
    for qa in m["qa_log"]:
        turns.append(RefinementTurn(role="assistant", content=qa["question"]))
        if qa.get("answer"):
            turns.append(RefinementTurn(role="user", content=qa["answer"]))
    return turns


def _refinement_generate_next_batch(m: dict) -> None:
    """Call the generator once and append the new questions to qa_log."""
    refined: RefinedPrompt = m["refined"]
    _, output = advance_turn(
        original_prompt=m["user_prompt"],
        refined_prompt=refined.refined,
        intent=m["intent"],
        profiles=m["bundle"],
        retrieval=m["retrieval"],
        prior_turns=_refinement_prior_turns(m),
        api_key=st.session_state["intent_api_key"] or None,
        model=st.session_state["intent_model"] or None,
    )
    m["last_proposed"] = output.proposed_prompt
    m["generator_done"] = output.done
    for q in output.questions:
        m["qa_log"].append(
            {"question": q.question, "choices": list(q.choices), "answer": None}
        )


def _record_refinement_answer(m: dict, qa_idx: int, answer: str) -> None:
    """Persist an answer to qa_log[qa_idx], advance, and queue next batch if needed."""
    answer = answer.strip()
    if not answer:
        return
    qa = m["qa_log"][qa_idx]
    qa["answer"] = answer
    qa["skipped"] = False
    m["refined"] = append_qa(m["refined"], qa["question"], answer)
    m["pending_idx"] = qa_idx + 1
    if m["refined"].turns_count >= MAX_REFINEMENT_TURNS:
        return
    # If we've answered every question we have, ask the generator for more.
    if m["pending_idx"] >= len(m["qa_log"]) and not m.get("generator_done"):
        _refinement_generate_next_batch(m)


def _skip_refinement_question(m: dict, qa_idx: int) -> None:
    """Mark a single question as skipped and advance to the next pending one.

    Unlike :func:`_record_refinement_answer`, this does NOT fold anything
    into the refined prompt and does NOT count toward the iteration cap —
    the user simply declined this clarifier.
    """
    qa = m["qa_log"][qa_idx]
    qa["answer"] = None
    qa["skipped"] = True
    m["pending_idx"] = qa_idx + 1
    if m["refined"].turns_count >= MAX_REFINEMENT_TURNS:
        return
    if m["pending_idx"] >= len(m["qa_log"]) and not m.get("generator_done"):
        _refinement_generate_next_batch(m)


def _render_refined_prompt_footer(idx: int, m: dict) -> None:
    refined: RefinedPrompt = m["refined"]
    locked = m.get("locked", False)
    skipped = m.get("skipped", False)

    with st.chat_message("assistant", avatar="📋"):
        st.markdown(
            _substep_chip(
                "Stage 4 · Current refined prompt",
                bg="#fef3c7", fg="#92400e", kind="footer",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(f"> {refined.refined.replace(chr(10), chr(10) + '> ')}")
        st.caption(
            f"Clarifications added: {refined.turns_count} / {MAX_REFINEMENT_TURNS}"
        )

        if locked:
            pivot = m.get("pivot", False)
            resolved = m.get("pivot_resolved", False)

            if not pivot:
                st.success(
                    "Locked in. Refined prompt stored; existing snippets reused "
                    "(no intent pivot)."
                )
                _render_generate_button(idx, m)
                return

            if resolved:
                decision = m.get("pivot_decision", "")
                if decision == "regenerate":
                    st.success(
                        "Locked in. Intent pivoted — new profiles + retrieval "
                        "appended below. Continue through the new stages; "
                        "Stage 5 will appear after the next refinement is "
                        "locked in."
                    )
                    # Intentionally no generate button here: this refinement's
                    # snippets/profile bundle have been superseded by the
                    # regenerated pipeline below. The new refinement message
                    # owns the Proceed → Generate content button.
                else:
                    st.info(
                        "Locked in. Intent pivoted but you chose to keep the "
                        "existing snippets."
                    )
                    _render_generate_button(idx, m)
                return

            # Pivot detected, awaiting the user's choice.
            new_intent: IntentResult = m["new_intent"]
            orig_intent: IntentResult = m["intent"]
            st.warning(
                "**Intent pivoted on the refined prompt.** "
                "Old → new tasks: "
                f"`{orig_intent.task_codes}` → `{new_intent.task_codes}`; "
                f"sector `{orig_intent.sector!r}` → `{new_intent.sector!r}`; "
                f"keywords `{orig_intent.topic_keywords}` → "
                f"`{new_intent.topic_keywords}`."
            )
            regen_col, keep_col = st.columns([1, 1], vertical_alignment="bottom")
            with regen_col:
                regen = st.button(
                    "Regenerate (new profiles + retrieval)",
                    key=f"refine_regen_{idx}",
                    type="primary",
                    use_container_width=True,
                )
            with keep_col:
                keep = st.button(
                    "Keep current snippets",
                    key=f"refine_keep_{idx}",
                    use_container_width=True,
                )
            if regen:
                _apply_pivot_regenerate(m)
                st.rerun()
            elif keep:
                _apply_pivot_keep(m)
                st.rerun()
            return
        if skipped:
            st.info(
                "Refinement skipped. Using the prompt as-is with the existing "
                "snippets."
            )
            _render_generate_button(idx, m)
            return

        col_lock, col_skip = st.columns([1, 1], vertical_alignment="bottom")
        with col_lock:
            lock = st.button(
                "Use this prompt",
                key=f"refine_lock_{idx}",
                type="primary",
                use_container_width=True,
            )
        with col_skip:
            skip = st.button(
                "Skip refinement",
                key=f"refine_skip_{idx}",
                use_container_width=True,
            )
        if lock:
            _commit_refinement(m)
            st.rerun()
        elif skip:
            m["skipped"] = True
            st.rerun()


def _render_generate_button(idx: int, m: dict) -> None:
    """Stage 4 → Stage 5 hand-off button. Rendered inside the locked footer
    so the action sits right next to the locked-in refined prompt."""
    if m.get("generated"):
        st.caption("Content generated below ↓")
        return
    if st.button(
        "Proceed → Generate content",
        key=f"content_proceed_{idx}",
        type="primary",
    ):
        _proceed_to_generation(idx, m)
        st.rerun()


def _render_refinement_question(idx: int, m: dict, qa_idx: int) -> None:
    qa = m["qa_log"][qa_idx]
    pending = qa_idx == m["pending_idx"] and not m.get("locked") and not m.get("skipped")
    total = len(m["qa_log"])

    with st.chat_message("assistant", avatar="❓"):
        st.markdown(
            _substep_chip(f"Q{qa_idx + 1} of {total}"),
            unsafe_allow_html=True,
        )
        st.markdown(f"**{qa['question']}**")

        # Inline answer rendering: the user's reply lives in the question
        # bubble itself (no separate user bubble) so the dialog reads as
        # Q + nested A rather than alternating avatars.
        if qa.get("answer"):
            st.markdown(
                f'<div style="margin-top:0.5rem;margin-bottom:0.75rem;'
                f'padding:8px 12px;'
                f'background:#ffffff80;border-left:3px solid #16a34a;'
                f'border-radius:6px;">'
                f'<span style="color:#15803d;font-weight:600;font-size:0.8em;">'
                f'✅ Your answer</span><br>'
                f'<span style="color:#1f2937;">{qa["answer"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif qa.get("skipped"):
            st.markdown(
                '<div style="margin-top:0.5rem;margin-bottom:0.75rem;'
                'padding:8px 12px;'
                'background:#ffffff80;border-left:3px solid #94a3b8;'
                'border-radius:6px;">'
                '<span style="color:#475569;font-weight:600;font-size:0.8em;">'
                '⏭ Skipped</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        if not pending:
            return

        if m["refined"].turns_count >= MAX_REFINEMENT_TURNS:
            st.caption("Iteration cap reached — lock in or skip below.")
            return

        choices: list[str] = qa.get("choices") or []
        if choices:
            cols = st.columns(min(len(choices), 4))
            for ci, choice in enumerate(choices):
                with cols[ci % len(cols)]:
                    if st.button(
                        choice,
                        key=f"refine_choice_{idx}_{qa_idx}_{ci}",
                        use_container_width=True,
                    ):
                        _record_refinement_answer(m, qa_idx, choice)
                        st.rerun()

        custom = st.text_input(
            "Or type your own answer",
            key=f"refine_custom_{idx}_{qa_idx}",
            placeholder="(free text; submit overrides the choice buttons above)",
        )
        submit_col, skip_col = st.columns([2, 1], vertical_alignment="bottom")
        with submit_col:
            submit = st.button(
                "Submit custom answer",
                key=f"refine_submit_{idx}_{qa_idx}",
                disabled=not custom or not custom.strip(),
                use_container_width=True,
            )
        with skip_col:
            skip_q = st.button(
                "Skip this question",
                key=f"refine_skip_q_{idx}_{qa_idx}",
                use_container_width=True,
            )
        if submit:
            _record_refinement_answer(m, qa_idx, custom.strip())
            st.rerun()
        elif skip_q:
            _skip_refinement_question(m, qa_idx)
            st.rerun()


def _render_refinement_message(idx: int, m: dict) -> None:
    with st.chat_message("assistant", avatar=_STAGE_STYLES["refinement"][0]):
        st.markdown(
            _block_anchor("refinement", idx) + _stage_chip("refinement"),
            unsafe_allow_html=True,
        )
        st.markdown("**Refining your prompt** — answer each question or skip below.")

    for qa_idx, _ in enumerate(m["qa_log"]):
        _render_refinement_question(idx, m, qa_idx)

    _render_refined_prompt_footer(idx, m)


# ── Stage 5: content generation ────────────────────────────────────────────


def _resolve_content_intent_bundle(m: dict) -> tuple[IntentResult, ProfileBundle]:
    """Pick the right intent + profile bundle to hand to the generator.

    If the user pivoted on the refined prompt and chose Regenerate, the
    pivot path appends new intent/profiles messages downstream and updates
    ``m["new_intent"]``; use that pair so the content reflects the latest
    user direction. Otherwise stick with the original intent + bundle that
    seeded refinement.
    """
    if m.get("pivot_decision") == "regenerate" and m.get("new_intent"):
        intent_for_gen: IntentResult = m["new_intent"]
        # The regenerate path appended a fresh profiles message; the most
        # recent one in chat history is the source of truth.
        bundle_for_gen: ProfileBundle = m["bundle"]
        for hist in reversed(st.session_state.messages):
            if hist.get("kind") == "profiles" and hist.get("intent") is intent_for_gen:
                bundle_for_gen = hist["bundle"]
                break
        return intent_for_gen, bundle_for_gen
    return m["intent"], m["bundle"]


def _proceed_to_generation(idx: int, m: dict) -> None:
    """Run Stage 5 and append a content message. ``idx`` is the refinement
    message index, used only for spinner / key uniqueness."""
    intent_for_gen, bundle_for_gen = _resolve_content_intent_bundle(m)
    refined_text = m.get("refined_text") or m["refined"].refined
    req = ContentRequest(
        refined_prompt=refined_text,
        intent=intent_for_gen,
        profiles=bundle_for_gen,
        snippets=m["retrieval"].snippets,
    )
    api_key = st.session_state.get("content_api_key") or None
    model = st.session_state.get("content_model") or None
    spinner_label = (
        f"Generating content via `{model}`…"
        if api_key and model
        else "Generating content (stub — set Content key in Settings)…"
    )
    with st.spinner(spinner_label):
        result = generate_content(req, api_key=api_key, model=model)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "content",
            "result": result,
            "refined_prompt": refined_text,
            "snippets": m["retrieval"].snippets,
            "source_refinement_idx": idx,
        }
    )
    m["generated"] = True

    # Stage 6: evaluation — auto-runs against the generation we just produced.
    eval_api_key = st.session_state.get("eval_api_key") or None
    eval_model = st.session_state.get("eval_model") or None
    strict_mode = bool(st.session_state.get("eval_strict_mode", False))
    eval_spinner = (
        f"Evaluating against KPI catalogue via `{eval_model}`…"
        if eval_api_key and eval_model
        else "Evaluating against KPI catalogue (deterministic only — "
        "set Evaluation key in Settings to enable LLM judges)…"
    )
    with st.spinner(eval_spinner):
        eval_result = evaluate(
            req,
            result,
            channel="web",
            origin="instant",
            api_key=eval_api_key,
            model=eval_model,
            strict_mode=strict_mode,
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "evaluation",
            "result": eval_result,
            "source_content_idx": len(st.session_state.messages) - 1,
        }
    )


def _render_content_message(idx: int, m: dict) -> None:
    result: ContentResult = m["result"]
    with st.chat_message("assistant", avatar=_STAGE_STYLES["content"][0]):
        st.markdown(
            _block_anchor("content", idx) + _stage_chip("content"),
            unsafe_allow_html=True,
        )
        model_caption = (
            f"_Generated via `{result.model}`._"
            if result.source == "llm" and result.model
            else "_Deterministic stub — configure Content key + model in Settings._"
        )
        st.markdown(model_caption)
        st.markdown(result.body)
        if result.citations:
            with st.expander(f"Sources ({len(result.citations)})"):
                snippet_by_index = {
                    i + 1: s for i, s in enumerate(m.get("snippets", []))
                }
                for c in result.citations:
                    snippet = snippet_by_index.get(c.index)
                    locator = f"`{c.source_doc}::{c.node_id}`"
                    line = (
                        f"**[{c.index}]** {c.title} — {locator}"
                        if snippet is None
                        else f"**[{c.index}]** {snippet.title} — {locator}  \n"
                        f"_score={snippet.score:.2f} · {snippet.reason}_"
                    )
                    st.markdown(line)


# ── Stage 6: evaluation ────────────────────────────────────────────────────


_MATURITY_DOT = {"high": "🟢", "medium": "🟡", "low": "🔴"}
_WEIGHT_SORT_KEY = {"Blocking": 0, "High": 1, "Medium": 2, "Low": 3}


def _verdict_banner(result: EvaluationResult) -> str:
    if result.passed:
        return (
            '<div style="background:#dcfce7;border-left:4px solid #166534;'
            'padding:10px 14px;border-radius:6px;font-weight:600;color:#14532d;">'
            "✅ Passed — no blocking KPI violations detected.</div>"
        )
    failed_list = ", ".join(f"`{kpi_id}`" for kpi_id in result.failed_blocking)
    return (
        '<div style="background:#fee2e2;border-left:4px solid #b91c1c;'
        'padding:10px 14px;border-radius:6px;font-weight:600;color:#7f1d1d;">'
        f"⛔ Blocked — {len(result.failed_blocking)} blocking KPI failure(s): "
        f"{failed_list}</div>"
    )


def _format_kpi_row(r: KPIResult) -> str:
    badge = "✅" if r.passed else ("⏭️" if r.source == "skipped" else "❌")
    weight_chip = (
        f'<span style="background:#fee2e2;color:#7f1d1d;padding:1px 6px;'
        f'border-radius:4px;font-size:0.78em;font-weight:600;">BLOCKING</span> '
        if r.weight == "Blocking"
        else ""
    )
    indicator = f"`{r.indicator}`" if r.indicator else "—"
    reason = f"  \n  _{r.reason}_" if r.reason else ""
    return (
        f"- {badge} {weight_chip}**{r.name}**  · indicator={indicator} · "
        f"value=`{r.value}`{reason}"
    )


def _render_evaluation_message(idx: int, m: dict) -> None:
    result: EvaluationResult = m["result"]
    with st.chat_message("assistant", avatar=_STAGE_STYLES["evaluation"][0]):
        st.markdown(
            _block_anchor("evaluation", idx) + _stage_chip("evaluation"),
            unsafe_allow_html=True,
        )
        caption = (
            f"_Evaluated via `{result.model}` (channel={result.channel}, "
            f"origin={result.origin})._"
            if result.source == "llm" and result.model
            else (
                "_Deterministic evaluation only — configure Evaluation key + "
                f"model in Settings to enable LLM judges (channel={result.channel}, "
                f"origin={result.origin})._"
            )
        )
        st.markdown(caption)
        st.markdown(_verdict_banner(result), unsafe_allow_html=True)

        # Per-category maturity dots.
        if result.maturity_by_category:
            cat_lines: list[str] = ["**Maturity by category**  "]
            for cat, level in sorted(result.maturity_by_category.items()):
                dot = _MATURITY_DOT.get(level, "⚪")
                cat_lines.append(f"- {dot} **{cat}** — {level}")
            st.markdown("\n".join(cat_lines))

        # Required editorial signoffs.
        if result.dclp_steps_required:
            st.markdown(
                "**Editorial signoff required (dCLP)**  \n"
                + "\n".join(f"- ⏳ `{s}`" for s in result.dclp_steps_required)
            )

        # Group results by tier for the breakdown.
        by_tier: dict[int, list[KPIResult]] = {1: [], 2: [], 3: []}
        for r in result.results:
            by_tier.setdefault(r.tier, []).append(r)

        with st.expander(
            f"Detailed KPI breakdown ({len(result.results)} checks)"
        ):
            tier_labels = {
                1: "Tier 1 — deterministic checks",
                2: "Tier 2 — LLM-judge rubrics",
                3: "Tier 3 — human signoff (dCLP)",
            }
            for tier, label in tier_labels.items():
                kpis = by_tier.get(tier) or []
                if not kpis:
                    continue
                # Stable sort: Blocking → High → Medium → Low; within weight,
                # failures first so the eye lands on what's wrong.
                kpis = sorted(
                    kpis,
                    key=lambda r: (
                        _WEIGHT_SORT_KEY.get(r.weight, 9),
                        0 if not r.passed else 1,
                        r.kpi_id,
                    ),
                )
                passing = sum(1 for r in kpis if r.passed)
                st.markdown(f"**{label}** — {passing}/{len(kpis)} passing")
                st.markdown("\n".join(_format_kpi_row(r) for r in kpis))


# ── Sidebar ────────────────────────────────────────────────────────────────


# One label per message ``kind``. ``user`` is included so the timeline also
# carries the prompt rows — the user can use them as natural section breaks
# when scrolling. Unknown kinds fall back to a generic ``Block`` label.
_TIMELINE_LABELS: dict[str, str] = {
    "user":       "Prompt",
    "intent":     "Intent",
    "profiles":   "Profiles",
    "retrieval":  "Retrieval",
    "refinement": "Refinement",
    "content":    "Content",
    "evaluation": "Evaluation",
}


def _timeline_items() -> list[dict]:
    """One timeline item per message block, top-to-bottom."""
    items: list[dict] = []
    for i, m in enumerate(st.session_state.messages):
        kind = "user" if m.get("role") == "user" else m.get("kind")
        if not kind:
            continue
        items.append(
            {
                "idx": i,
                "kind": kind,
                "label": _TIMELINE_LABELS.get(kind, kind.title()),
            }
        )
    return items


def _render_pipeline_sidebar() -> None:
    """Dynamic vertical timeline: one dot per message block.

    Each click jumps to the anchor at the top of the matching bubble. A
    small companion script (see :func:`_inject_scrollspy`) bolds the
    timeline item whose anchor is currently in the viewport.
    """
    items = _timeline_items()
    if not items:
        st.caption(
            "Phases will appear here as you progress through the pipeline."
        )
        return

    parts: list[str] = [
        '<div data-pipeline-bar style="padding:4px 2px 12px 2px;">',
        '<div style="font-size:0.72em;color:#6b7280;letter-spacing:0.08em;'
        'font-weight:700;text-transform:uppercase;margin:0 0 12px 6px;">'
        "Phases</div>",
        '<div style="position:relative;padding:2px 0 2px 18px;">',
        # Thin vertical track behind the dots.
        '<div style="position:absolute;left:6px;top:6px;bottom:6px;'
        'width:1px;background:#e5e7eb;"></div>',
    ]
    for it in items:
        target = f"block-{it['kind']}-{it['idx']}"
        parts.append(
            f'<a href="#{target}" data-timeline-target="{target}" '
            f'style="display:flex;align-items:center;gap:10px;'
            f'padding:4px 6px;margin:1px 0;border-radius:4px;'
            f'text-decoration:none;color:#6b7280;position:relative;" '
            f"onmouseover=\"this.style.background='#f3f4f6'\" "
            f"onmouseout=\"this.style.background='transparent'\">"
            f'<span style="position:absolute;left:-13px;width:7px;height:7px;'
            f'border-radius:50%;background:#9ca3af;border:2px solid #ffffff;'
            f'box-sizing:content-box;"></span>'
            f'<span data-timeline-label '
            f'style="font-size:0.88em;font-weight:400;">'
            f"{it['label']}</span></a>"
        )
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _inject_scrollspy() -> None:
    """Bold the timeline item whose section the user is currently in.

    Uses a deterministic *scroll-position* rule rather than
    IntersectionObserver visibility: the active anchor is the last one
    whose top has scrolled above a fixed offset from the viewport top.
    This avoids the flip-flop you get with IO when two zero-height
    anchors briefly co-exist in the same active band.

    Injected via ``components.html`` (Streamlit strips ``<script>`` from
    ``st.markdown``). A nonce derived from ``len(messages)`` is embedded
    so the iframe re-runs whenever a new block appears — otherwise the
    new anchor would never be picked up.
    """
    nonce = len(st.session_state.messages)
    script = (
        f"<!-- aurora-scrollspy v={nonce} -->\n"
        + """<script>
        (function () {
          const pdoc = window.parent.document;
          const pwin = window.parent;

          // Detach any listeners left over from the previous rerun.
          if (pdoc.__auroraTeardown) { pdoc.__auroraTeardown(); }
          pdoc.__auroraSuppressUntil = 0;

          // Offset (px) from the viewport top that defines "I am here".
          // An anchor counts as "passed" once its top has scrolled above
          // this line, and the most-recently-passed anchor is active.
          const TOP_OFFSET = 120;

          // Active-state colour per block kind — mirrors the bubble tint
          // colours defined in _STAGE_STYLES (Python side). Keep both in
          // sync if a stage's accent ever changes.
          const COLOR_BY_KIND = {
            user:       '#374151',
            intent:     '#1e40af',
            profiles:   '#166534',
            retrieval:  '#5b21b6',
            refinement: '#92400e',
            content:    '#0f172a',
            evaluation: '#155e75',
          };
          function colorForTarget(target) {
            // target is like ``block-{kind}-{idx}`` — pull the kind out.
            if (!target) return '#111827';
            const parts = target.split('-');
            return (parts.length >= 2 && COLOR_BY_KIND[parts[1]])
              || '#111827';
          }

          function applyStyles(activeKey, links) {
            const activeColor = colorForTarget(activeKey);
            links.forEach(a => {
              const k = a.getAttribute('data-timeline-target');
              const label = a.querySelector('[data-timeline-label]');
              const dot = a.querySelector('span');
              const active = (k === activeKey);
              if (label) {
                label.style.fontWeight = active ? '700' : '400';
                label.style.color = active ? activeColor : '#6b7280';
              }
              if (dot) {
                dot.style.background = active ? activeColor : '#9ca3af';
                dot.style.transform = active ? 'scale(1.25)' : 'none';
                dot.style.transition = 'all 0.15s ease';
              }
            });
          }

          function findActiveKey(anchors) {
            // Anchors are in DOM (= page) order. Walk top→bottom and
            // remember the LAST anchor whose top is at or above the
            // offset line. That's deterministic and monotonic during
            // scroll, so the highlight can't bounce between siblings.
            let active = null;
            for (const a of anchors) {
              if (a.getBoundingClientRect().top <= TOP_OFFSET) {
                active = a;
              } else {
                break;
              }
            }
            // Before the very first anchor (user scrolled above the
            // first block) fall back to the first one so something is
            // always highlighted.
            if (!active && anchors.length) active = anchors[0];
            return active ? active.getAttribute('data-block-anchor') : null;
          }

          let lastKey = null;
          let rafQueued = false;
          let anchorsCache = [];
          let linksCache = [];

          function repaint() {
            if (Date.now() < (pdoc.__auroraSuppressUntil || 0)) return;
            if (!anchorsCache.length || !linksCache.length) return;
            const key = findActiveKey(anchorsCache);
            if (key === lastKey) return;  // no-op when unchanged
            lastKey = key;
            applyStyles(key, linksCache);
          }

          function scheduleRepaint() {
            if (rafQueued) return;
            rafQueued = true;
            pwin.requestAnimationFrame(() => {
              rafQueued = false;
              repaint();
            });
          }

          // Streamlit lays the main content out under either
          // ``section[data-testid="stMain"]`` or directly on the
          // window — listen to scroll on both, plus the document with
          // capture so any inner scroller propagates.
          const scrollTargets = [pwin, pdoc];
          const stMain = pdoc.querySelector('section[data-testid="stMain"]')
                       || pdoc.querySelector('section.main')
                       || pdoc.querySelector('main');
          if (stMain) scrollTargets.push(stMain);

          function onScroll() { scheduleRepaint(); }
          scrollTargets.forEach(t =>
            t.addEventListener('scroll', onScroll, { passive: true, capture: true })
          );

          // Click → instant highlight, plus a short suppression window
          // so the observer doesn't overwrite the chosen item while the
          // browser is still mid-scroll.
          function bindClicks(links) {
            links.forEach(a => {
              if (a.__auroraClickAttached) return;
              a.addEventListener('click', () => {
                const k = a.getAttribute('data-timeline-target');
                lastKey = k;
                applyStyles(k, links);
                pdoc.__auroraSuppressUntil = Date.now() + 900;
              });
              a.__auroraClickAttached = true;
            });
          }

          function refresh() {
            const anchors = Array.from(
              pdoc.querySelectorAll('[data-block-anchor]')
            );
            const links = Array.from(
              pdoc.querySelectorAll('a[data-timeline-target]')
            );
            if (!anchors.length || !links.length) return false;
            anchorsCache = anchors;
            linksCache = links;
            bindClicks(links);
            // Default to the LAST anchor on first paint: Streamlit
            // auto-scrolls to the bottom when a new block lands, so
            // the user is almost certainly looking at the newest one.
            lastKey = anchors[anchors.length - 1]
              .getAttribute('data-block-anchor');
            applyStyles(lastKey, links);
            // One pass with the real scroll position in case the user
            // is already mid-page.
            scheduleRepaint();
            return true;
          }

          let tries = 0;
          const initTimer = pwin.setInterval(() => {
            if (refresh() || ++tries > 30) pwin.clearInterval(initTimer);
          }, 100);

          pdoc.__auroraTeardown = function () {
            scrollTargets.forEach(t =>
              t.removeEventListener('scroll', onScroll, { capture: true })
            );
            pwin.clearInterval(initTimer);
          };
        })();
        </script>"""
    )
    components.html(script, height=0)


with st.sidebar:
    _render_pipeline_sidebar()
    st.divider()
    if st.button(
        "Clear conversation",
        disabled=not st.session_state.messages,
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

# Scroll-spy lives outside the sidebar block so its iframe doesn't get
# rendered inside the sidebar column (it has 0 height either way, but
# keeping it at the page level avoids weird flex layout interactions).
_inject_scrollspy()


# ── Replay loop ────────────────────────────────────────────────────────────


messages = st.session_state.messages


def _latest_idx_of(kind: str) -> int | None:
    return next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("kind") == kind),
        None,
    )


latest_intent_idx = _latest_idx_of("intent")
latest_profiles_idx = _latest_idx_of("profiles")
latest_retrieval_idx = _latest_idx_of("retrieval")
for idx, m in enumerate(messages):
    if m["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(_block_anchor("user", idx), unsafe_allow_html=True)
            st.markdown(m["content"])
    elif m.get("kind") == "intent":
        _render_intent_message(idx, m, is_latest_intent=(idx == latest_intent_idx))
    elif m.get("kind") == "profiles":
        _render_profiles_message(idx, m, is_latest_profiles=(idx == latest_profiles_idx))
    elif m.get("kind") == "retrieval":
        _render_retrieval_message(idx, m, is_latest_retrieval=(idx == latest_retrieval_idx))
    elif m.get("kind") == "refinement":
        _render_refinement_message(idx, m)
    elif m.get("kind") == "content":
        _render_content_message(idx, m)
    elif m.get("kind") == "evaluation":
        _render_evaluation_message(idx, m)
    else:
        with st.chat_message(m["role"]):
            st.markdown(m.get("content", ""))


# ── New prompt ─────────────────────────────────────────────────────────────


prompt = st.chat_input("Ask AURORA…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    spinner_label = (
        "Classifying intent via LLM…"
        if st.session_state["intent_api_key"] and st.session_state["intent_model"]
        else "Classifying intent (deterministic fallback)…"
    )
    with st.spinner(spinner_label):
        result, source = classify_full(
            prompt,
            api_key=st.session_state["intent_api_key"] or None,
            model=st.session_state["intent_model"] or None,
        )
    raw = result.model_dump_json(indent=2) if source == "llm" else None
    msg = {
        "role": "assistant",
        "kind": "intent",
        "intent": result,
        "source": source,
        "raw": raw,
        "user_prompt": prompt,
        "proceeded": False,
    }
    st.session_state.messages.append(msg)
    _render_intent_message(
        len(st.session_state.messages) - 1, msg, is_latest_intent=True
    )
