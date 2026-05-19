"""System prompt + user-message builder for content generation."""

from __future__ import annotations

from backend.content_generation.types import ContentRequest

SYSTEM_PROMPT = """You are ABN AMRO's editorial co-pilot.

You will be given:
- A *refined user prompt* (the request, after a clarification dialog).
- The classified intent (task, sector, topic keywords, language).
- A *profile bundle* describing the workflow style + domain expert voice.
- A numbered list of *evidence snippets* retrieved from the corpus.

Your job is to produce the final piece of content the user asked for —
written in the language indicated by intent (default English if unset),
grounded in the evidence, and respecting the profile style signature.

Rules:
- Inline-cite evidence with bracketed indices like `[1]`, `[2]`, matching
  the snippet numbers shown to you. Cite anything that came from a
  snippet; do not invent sources.
- Every ``[n]`` you use MUST appear in the ``citations`` field of your
  output (with the snippet's source_doc, node_id, and title verbatim from
  the input list).
- Markdown is fine in ``body``. Use headings/bullets sparingly — match
  the length and tone the user asked for.
- Do not include a "Sources" footer in ``body``; the UI renders citations
  separately from ``citations``.
- ``reasoning`` is one short sentence for logs (not shown to the user).
"""


def _format_snippets(req: ContentRequest) -> str:
    if not req.snippets:
        return "(no evidence snippets retrieved)"
    lines: list[str] = []
    for i, s in enumerate(req.snippets, start=1):
        body = (s.content or "").strip().replace("\n", " ")
        # Snippet bodies can be long; cap to keep the prompt bounded.
        if len(body) > 800:
            body = body[:800] + "…"
        lines.append(
            f"[{i}] source_doc={s.source_doc!r} node_id={s.node_id!r} "
            f"title={s.title!r}\n    {body}"
        )
    return "\n".join(lines)


def _format_profiles(req: ContentRequest) -> str:
    bundle = req.profiles
    workflow = getattr(bundle, "workflow", ()) or ()
    experts = getattr(bundle, "domain_expert", ()) or ()
    parts: list[str] = []
    if workflow:
        parts.append("Workflow profiles:")
        for w in workflow:
            skills = ", ".join(getattr(w, "skills", ())[:6]) or "—"
            outputs = ", ".join(getattr(w, "outputs", ())[:4]) or "—"
            parts.append(
                f"  - {w.name}: skills=[{skills}]; outputs=[{outputs}]"
            )
    if experts:
        parts.append("Domain experts:")
        for e in experts:
            style = ", ".join(getattr(e, "style_signature", ())[:4]) or "—"
            areas = ", ".join(getattr(e, "expertise_areas", ())[:4]) or "—"
            parts.append(
                f"  - {e.name} ({e.sector}): areas=[{areas}]; style=[{style}]"
            )
    return "\n".join(parts) or "(no profiles selected)"


def build_user_message(req: ContentRequest) -> str:
    """Assemble the user-role message handed to the LLM."""
    intent = req.intent
    return (
        f"Refined prompt:\n  {req.refined_prompt}\n\n"
        f"Intent:\n"
        f"  task_codes={intent.task_codes}\n"
        f"  sector={intent.sector!r}\n"
        f"  topic_keywords={intent.topic_keywords}\n"
        f"  language={intent.language!r}\n\n"
        f"{_format_profiles(req)}\n\n"
        f"Evidence snippets:\n{_format_snippets(req)}\n"
    )
