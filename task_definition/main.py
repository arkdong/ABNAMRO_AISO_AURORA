import streamlit as st
import json
from datetime import datetime, date

from config import WRITING_GUIDE, ROLES, CHANNEL_PROFILES, DOCUMENT_REGISTRY, TASK_LABELS
from corpus_manager import CorpusManager
from intent_classifier import IntentClassifier
from prompt_builder import EchoPromptBuilder
from loguru import logger

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURORA – Editorial Co-pilot POC", page_icon="🏦", layout="wide"
)

st.markdown(
    """
<style>
  .role-badge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.85rem; margin-bottom:8px; }
  .tag-pill   { display:inline-block; background:#e8f0fe; color:#1a56db; border-radius:12px; padding:2px 10px; font-size:0.78rem; margin:2px; }
  .hard-rule  { background:#fef3c7; border-left:4px solid #f59e0b; padding:6px 12px; border-radius:4px; margin-bottom:4px; font-size:0.84rem; }
  .soft-rule  { background:#ecfdf5; border-left:4px solid #10b981; padding:6px 12px; border-radius:4px; margin-bottom:4px; font-size:0.84rem; }
  .echo-box   { background:#1e1e2e; color:#cdd6f4; border-radius:8px; padding:16px; font-family:monospace; font-size:0.80rem; white-space:pre-wrap; line-height:1.6; }
  .flow-step  { background:white; border:1px solid #e5e7eb; border-radius:8px; padding:9px 14px; margin-bottom:5px; display:flex; align-items:center; gap:10px; font-size:0.84rem; }
  .check-pass { color:#059669; font-weight:600; }
  .check-fail { color:#dc2626; font-weight:600; }
  .check-warn { color:#d97706; font-weight:600; }
  .section-header { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#6b7280; margin-bottom:6px; }
  .confidence-bar-wrap { background:#e5e7eb; border-radius:8px; height:10px; width:100%; margin-top:4px; margin-bottom:10px; }
  .article-card { background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
  .article-card h4 { margin:0 0 4px 0; font-size:0.9rem; color:#111827; }
  .article-card p  { margin:0; font-size:0.80rem; color:#6b7280; }
  .checklist-row   { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #f3f4f6; font-size:0.84rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Initialize core modules ───────────────────────────────────────────────────
@st.cache_resource
def get_corpus_manager():
    return CorpusManager()

corpus_manager = get_corpus_manager()
classifier = IntentClassifier()
prompt_builder = EchoPromptBuilder()


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/ABN_AMRO_logo.svg/250px-ABN_AMRO_logo.svg.png",
        width=140,
    )
    st.title("AURORA")
    st.caption(
        "ABN AMRO AISO Editorial Co-pilot\nTask Definition & Dataflow POC · Stage 1"
    )
    st.divider()
    api_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="If provided, enables LLM-based intent classification.",
    )
    st.divider()
    lang_out = st.selectbox("Output language", ["English", "Dutch", "Both (EN + NL)"])
    retrieval_track = st.selectbox(
        "Retrieval track",
        [
            "Hybrid (PageIndex guides + Vector examples)",
            "Track B – PageIndex only",
            "Track A – Vector RAG only",
        ],
    )
    st.divider()
    st.markdown("**Test queries**")
    test_queries = {
        "🍽️ AI & food/restaurants": "I want to write an article about the impact of AI in food and restaurants. Does there exist any article about it or any related information?",
        "🌐 Translate article": "Help me translate this article into English using the writing reference: 'Technologie en voedselproductie: kansen voor de agrarische sector'",
        "📅 Aging articles": "Are there any old articles older than 2 years that do not align with the current writing reference?",
        "✅ Draft → checklist": "Take a draft that finally got approved and run the first draft through the application to see whether it will pass the checklist.",
    }
    for label, query in test_queries.items():
        if st.button(label, use_container_width=True):
            st.session_state["selected_query"] = query


# ── Main ─────────────────────────────────────────────────────────────────────────
st.title("🏦 AURORA — Editorial Co-pilot")
st.caption("Task Definition · Channel Dataflow · ECHO Prompt Assembly")
st.divider()

prefill = st.session_state.get("selected_query", "")
user_input = st.text_area(
    "Enter your request",
    value=prefill,
    height=90,
    placeholder="Try one of the test queries from the sidebar →",
)

run = st.button("⚡  Run pipeline", type="primary", disabled=not user_input.strip())

if user_input.strip() and run:
    logger.info(f"Pipeline started for request: '{user_input[:50]}...'")
    role, task_code, confidence, task_reason, raw_llm_output = classifier.classify(user_input, api_key)
    logger.info(f"Classified as {role} / {task_code} with {confidence:.2f} confidence")
    role_data = ROLES.get(role, ROLES["Insights Editorial"])
    conf_pct = int(confidence * 100)
    conf_color = (
        "#10b981" if conf_pct >= 80 else "#f59e0b" if conf_pct >= 60 else "#ef4444"
    )

    # ── Row 1: Classification ────────────────────────────────────────────────────
    st.subheader("① Classification")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
        <span class='role-badge' style='background:{role_data["bg"]};color:{role_data["color"]}'>
          {role_data["emoji"]} {role}
        </span><br>
        <span style='font-size:0.82rem;color:#374151'>{TASK_LABELS.get(task_code, "")}</span>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Confidence", f"{conf_pct}%")
        st.markdown(
            f"""<div class='confidence-bar-wrap'>
          <div style='background:{conf_color};height:10px;border-radius:8px;width:{conf_pct}%'></div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(f"**Reason:** {task_reason}")
        st.markdown(f"**Retrieval:** {retrieval_track}")
        st.caption("Router specialization: Insights Editorial profile active in Stage 1.")

    if raw_llm_output:
        with st.expander("🤖 Raw LLM Output"):
            st.code(raw_llm_output, language="json")

    st.divider()

    # ── Channel profile view (generic router vs specialisation) ──────────────────
    st.subheader("② Channel Profile & Guidelines")

    profile = CHANNEL_PROFILES.get(role, CHANNEL_PROFILES["Insights Editorial"])
    pc1, pc2 = st.columns([1.3, 1])

    with pc1:
        st.markdown(
            "<div class='section-header'>Active profile</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <p style="font-size:0.9rem;color:#111827;margin-bottom:4px;"><strong>{role}</strong></p>
            <p style="font-size:0.8rem;color:#4b5563;">{profile['description']}</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-header'>Hard rules snapshot</div>",
            unsafe_allow_html=True,
        )
        for r in profile["hard_rules"][:4]:
            st.markdown(f"<div class='hard-rule'>{r}</div>", unsafe_allow_html=True)

    with pc2:
        st.markdown(
            "<div class='section-header'>Soft guidance snapshot</div>",
            unsafe_allow_html=True,
        )
        for g in profile["soft_guidance"][:4]:
            st.markdown(f"<div class='soft-rule'>{g}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'>Skills</div>", unsafe_allow_html=True
        )
        pills = " ".join(
            f"<span class='tag-pill'>{s}</span>" for s in profile["skills"]
        )
        st.markdown(pills, unsafe_allow_html=True)

    with st.expander("🔎 View other channel profiles (Anna, App, Web/IB)"):
        cols = st.columns(3)
        for i, ch in enumerate(["Chatbot (Anna)", "Mobile App (UX)", "Web / IB"]):
            with cols[i]:
                p = CHANNEL_PROFILES.get(ch)
                if p:
                    st.markdown(
                        f"<strong style='font-size:0.85rem;'>{ch}</strong><br>"
                        f"<span style='font-size:0.75rem;color:#4b5563;'>{p['description']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div class='section-header' style='margin-top:6px;'>Hard rules</div>",
                        unsafe_allow_html=True,
                    )
                    for r in p["hard_rules"][:2]:
                        st.markdown(
                            f"<div class='hard-rule'>{r}</div>", unsafe_allow_html=True
                        )

    st.divider()

    # ── Conditional dataflow per task ────────────────────────────────────────────

    # ── T1_SEARCH ────────────────────────────────────────────────────────────────
    if task_code == "T1_SEARCH":
        st.subheader("③ Corpus Search — Prior art & related content")
        results = corpus_manager.search(user_input)
        if results:
            st.success(
                f"Found **{len(results)}** related article(s) in the Insights corpus."
            )
            for art in results:
                age_years = (date.today() - art["date"]).days // 365
                st.markdown(
                    f"""
                <div class='article-card'>
                  <h4>{art.get("emoji","📄")} {art["title"]}</h4>
                  <p>🗓 {art["date"].strftime("%B %Y")} · {art["lang"].upper()} · {"✅ Compliant" if art["compliant"] else "⚠️ Compliance issues"} · {age_years} year(s) old</p>
                  <p style="margin-top:6px">{art["excerpt"]}</p>
                  <p style="margin-top:4px;font-style:italic;color:#374151">{art["summary_en"]}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "No directly matching articles found. Proceeding to draft from Writing Guide only."
            )
        retrieved_docs = results

        st.divider()
        st.subheader("④ Next step — T1 Draft")
        st.info(
            "Based on the search results above, the system would now move to **T1 Draft** mode, "
            "pre-loading the related articles as retrieved context into the ECHO prompt."
        )
        if st.button("Preview ECHO Prompt Pack"):
            echo = prompt_builder.build(role, "T1_DRAFT", user_input, retrieved_docs, lang_out)
            st.markdown(f"<div class='echo-box'>{echo}</div>", unsafe_allow_html=True)

    # ── T1_TRANSLATE ─────────────────────────────────────────────────────────────
    elif task_code == "T1_TRANSLATE":
        st.subheader("③ Translation Dataflow")
        match = next(
            (
                a
                for a in corpus_manager.corpus
                if any(kw in user_input.lower() for kw in a["title"].lower().split())
            ),
            None,
        )
        if match:
            st.success(
                f"Source article identified: **{match['title']}** ({match['lang'].upper()}, {match['date'].year})"
            )
            st.markdown(
                f"""
            <div class='article-card'>
              <h4>📄 {match["title"]}</h4>
              <p>📅 {match["date"].strftime("%B %Y")} · {match["lang"].upper()} · {"✅ Compliant" if match["compliant"] else "⚠️ Has compliance issues"}</p>
              <p style="margin-top:6px">{match["excerpt"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            retrieved_docs = [match]
        else:
            st.warning(
                "Source article not identified by title — please provide the article text directly."
            )
            retrieved_docs = []

        st.markdown("**Writing Guide applied to translation:**")
        for g in WRITING_GUIDE["soft_guidance"][4:6]:
            st.markdown(f"<div class='soft-rule'>{g}</div>", unsafe_allow_html=True)

        st.divider()
        echo = prompt_builder.build(role, task_code, user_input, retrieved_docs, lang_out)
        st.subheader("④ ECHO Prompt Pack")
        st.markdown(f"<div class='echo-box'>{echo}</div>", unsafe_allow_html=True)

    # ── T4_RENEWAL ───────────────────────────────────────────────────────────────
    elif task_code == "T4_RENEWAL":
        st.subheader("③ Article Aging & Compliance Scan (T4)")
        aging = corpus_manager.find_aging_articles(years=2)
        st.markdown(f"**{len(aging)}** article(s) older than 2 years found in corpus.")

        for art in aging:
            issues, passes = corpus_manager.check_compliance(art)
            age_years = (date.today() - art["date"]).days // 365
            status_label = (
                f"⚠️ {len(issues)} compliance issue(s)" if issues else "✅ Compliant"
            )
            with st.expander(
                f"{'⚠️' if issues else '✅'} {art['title']} — {age_years}y old · {status_label}"
            ):
                st.markdown(
                    f"**Language:** {art['lang'].upper()}  |  **Published:** {art['date'].strftime('%B %Y')}"
                )
                st.markdown(f"**Summary:** {art['summary_en']}")
                if issues:
                    st.markdown("**Compliance issues detected:**")
                    for i in issues:
                        st.markdown(
                            f"<div class='hard-rule'>❌ {i}</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown(
                    "**Recommended action:** Retrieve current Writing Guide sections and propose a revised draft aligned with V1.1 guidelines."
                )

        st.divider()
        st.subheader("④ Renewal Pipeline")
        steps = [
            ("📅", "Aging articles identified (> 2 years old)"),
            ("📋", "Each article checked against Writing Guide 2026 V1.1"),
            (
                "🔍",
                "Retrieval: fetch relevant updated guideline sections (PageIndex / Vector)",
            ),
            (
                "🧩",
                "ECHO assembles: original article + updated guidelines + hard rules",
            ),
            ("🤖", "LLM proposes revised draft with tracked changes"),
            ("✅", "Compliance re-check on revised draft"),
            ("👤", "Human editor reviews and approves renewal"),
        ]
        for emoji, label in steps:
            st.markdown(
                f"<div class='flow-step'><div style='font-size:1.1rem'>{emoji}</div><div>{label}</div></div>",
                unsafe_allow_html=True,
            )

    # ── T2_COMPLIANCE ────────────────────────────────────────────────────────────
    elif task_code == "T2_COMPLIANCE":
        st.subheader("③ Compliance Check (T2) — Draft → Checklist")

        # Simulate: use a known non-compliant article as the "first draft" for demo
        if len(corpus_manager.corpus) >= 3:
            draft_article = corpus_manager.corpus[2]
        elif len(corpus_manager.corpus) > 0:
            draft_article = corpus_manager.corpus[0]
        else:
            draft_article = {"title": "No corpus", "compliance_issues": ["No rules passed"]}

        st.info(
            f"**Simulating:** Running the initial draft of **'{draft_article['title']}'** through the compliance pipeline."
        )

        issues, passes = corpus_manager.check_compliance(draft_article)

        st.markdown("### Compliance Checklist — Writing Guide 2026 V1.1")
        all_rules = WRITING_GUIDE["hard_rules"]
        for rule in all_rules:
            failed = any(rule[:18].lower() in i.lower() for i in issues)
            if failed:
                matching_issue = next(
                    (i for i in issues if rule[:18].lower() in i.lower()),
                    "Rule violation detected",
                )
                st.markdown(
                    f"<div class='checklist-row'><span class='check-fail'>❌ FAIL</span> "
                    f"<span>{rule}</span><br><small style='color:#9ca3af;margin-left:30px'>↳ {matching_issue}</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='checklist-row'><span class='check-pass'>✅ PASS</span> <span>{rule}</span></div>",
                    unsafe_allow_html=True,
                )

        total = len(all_rules)
        n_fail = len(issues)
        n_pass = total - n_fail
        st.divider()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rules checked", total)
        col_b.metric("Passed", n_pass, delta=None)
        col_c.metric(
            "Failed",
            n_fail,
            delta=f"-{n_fail}" if n_fail else None,
            delta_color="inverse",
        )

        if n_fail > 0:
            st.error(
                f"**Draft does NOT pass the checklist.** {n_fail} issue(s) must be resolved before CRM ingestion."
            )
            st.markdown(
                "**Recommended next step:** Route back to T1 Draft mode with flagged issues injected into the ECHO prompt for targeted revision."
            )
        else:
            st.success(
                "**Draft passes all hard rules.** Ready for human reviewer approval and CRM ingestion."
            )

        st.divider()
        echo = prompt_builder.build(role, task_code, user_input, [draft_article], lang_out)
        with st.expander("📋 ECHO Prompt Pack for revision"):
            st.markdown(f"<div class='echo-box'>{echo}</div>", unsafe_allow_html=True)

    # ── T1_DRAFT (default) ───────────────────────────────────────────────────────
    else:
        st.subheader("③ Retrieval — Related content")
        results = corpus_manager.search(user_input)
        if results:
            st.success(f"{len(results)} related article(s) retrieved from corpus.")
            for art in results[:2]:
                st.markdown(
                    f"""
                <div class='article-card'>
                  <h4>📄 {art["title"]}</h4>
                  <p>📅 {art["date"].strftime("%B %Y")} · {art["lang"].upper()} · {"✅ Compliant" if art["compliant"] else "⚠️ Compliance flags"}</p>
                  <p style="margin-top:5px">{art["summary_en"]}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "No related articles found — draft will be grounded in Writing Guide only."
            )
        retrieved_docs = results

        st.divider()
        st.subheader("④ Applied Rules")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(
                "<div class='section-header'>🔴 Hard Rules</div>",
                unsafe_allow_html=True,
            )
            for r in WRITING_GUIDE["hard_rules"][:4]:
                st.markdown(f"<div class='hard-rule'>{r}</div>", unsafe_allow_html=True)
        with rc2:
            st.markdown(
                "<div class='section-header'>🟢 Soft Guidance</div>",
                unsafe_allow_html=True,
            )
            for g in WRITING_GUIDE["soft_guidance"][:4]:
                st.markdown(f"<div class='soft-rule'>{g}</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("⑤ ECHO Prompt Pack")
        echo = prompt_builder.build(role, task_code, user_input, retrieved_docs, lang_out)
        st.markdown(f"<div class='echo-box'>{echo}</div>", unsafe_allow_html=True)

    # ── Task Definition JSON (always) ────────────────────────────────────────────
    st.divider()
    with st.expander("🗂 Full Task Definition JSON"):
        td = {
            "task_id": f"AURORA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "user_request": user_input,
            "classification": {
                "role": role,
                "task_type": task_code,
                "confidence": confidence if confidence <= 1 else confidence / 100,
                "reason": task_reason,
            },
            "retrieval_track": retrieval_track,
            "output_language": lang_out,
            "writing_guide": WRITING_GUIDE["version"],
            "channel_profiles_available": list(CHANNEL_PROFILES.keys()),
            "documents": list(DOCUMENT_REGISTRY.values()),
            "human_in_loop": True,
        }
        st.json(td)
        logger.success(f"Pipeline execution completed successfully for task_id: {td['task_id']}")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AURORA POC · ABN AMRO AISO · Stage 1 · May 2026")