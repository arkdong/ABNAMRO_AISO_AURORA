"""Prompts for Tier 2 LLM judges.

One shared system message, one anchor per rubric. The user message is
assembled by :func:`build_judge_user_message` from the request / generation
/ retrieved snippets.
"""

from __future__ import annotations

from backend.content_generation.types import ContentRequest, ContentResult

JUDGE_SYSTEM_PROMPT = """You are a content-quality auditor at ABN AMRO.

You will receive:
- A *user query* (what the customer asked for).
- A list of *retrieved evidence snippets* (the corpus context the writer had).
- A *generated content* draft.
- A *rubric* describing exactly one quality dimension to score.
- An *indicator scale* — the only allowed values for your score.

Your job: score the draft on this rubric only. Return:
- ``value``: one value from the indicator scale, verbatim.
- ``reason``: one short sentence (max 25 words) justifying the score.

Rules:
- Be strict but fair. Borderline cases default to the lower score.
- Do not invent claims about the draft that are not in it.
- The scale's allowed values are listed in the rubric; pick exactly one.
- If you cannot judge (e.g. draft is empty), return ``"unknown"``.
"""


RUBRICS: dict[str, str] = {
    "factuality": (
        "**Factuality.** Count substantive factual errors in the draft. "
        "A factual error is a claim that contradicts the retrieved evidence "
        "or is not supported by it at all. Misattribution and made-up "
        "numbers count; minor stylistic differences do not."
    ),
    "truthfullness": (
        "**Truthfulness.** Identify deviations from bank standards for "
        "truthful substance (e.g. using 'advice' for execution-only services, "
        "claiming guarantees the bank does not actually offer, dramatising "
        "risk/return). Count any deviation, however small."
    ),
    "relevancy": (
        "**Relevancy.** Does the draft address the user's actual query? "
        "Score by how directly the body responds to what the customer asked "
        "for, not by writing quality."
    ),
    "groundedness": (
        "**Groundedness.** What share of the draft's substantive claims can "
        "be traced back to the retrieved snippets? Full = every claim "
        "anchored; reasonable = most; limited = a few; none = none."
    ),
    "privacy_security": (
        "**Privacy & security.** Identify deviations from bank policy: "
        "exposing personal data, linking to non-AAB external domains, "
        "recommending insecure flows, mishandling BSN/IBAN, etc."
    ),
    "completeness_source": (
        "**Completeness (vs. source).** Does the draft cover the substance "
        "from the retrieved snippets that is relevant to the user's query? "
        "Score the degree of coverage of the relevant material."
    ),
    "comprehensiveness": (
        "**Comprehensiveness (answer).** Does the draft answer the user's "
        "*entire* query, including reasonable follow-up sub-questions a "
        "customer would have? AEO-style answer completeness."
    ),
    "clarity": (
        "**Clarity.** How clear is the substance of the draft? Count "
        "ambiguities — places where a reader could reasonably get a "
        "different meaning."
    ),
    "uniqueness_added_value": (
        "**Unique, added value.** Does the draft add value beyond restating "
        "the obvious or the snippet text? Reject pure boilerplate."
    ),
    "demonstrable_expertise": (
        "**Demonstrable expertise.** Does the draft reference concrete "
        "experience, examples, or expert customer-context detail (not "
        "generic marketing copy)?"
    ),
    "no_paraphrase": (
        "**No paraphrase.** Does the draft go beyond paraphrasing the "
        "retrieved snippets? A draft that just reshuffles sentences from "
        "the snippets fails this check."
    ),
    "no_filler": (
        "**No filler.** Is the draft free of filler/generic statements "
        "('It's important to consider…', 'In today's world…')?"
    ),
}


def _snippet_block(req: ContentRequest, max_chars: int = 600) -> str:
    if not req.snippets:
        return "(no evidence snippets)"
    lines: list[str] = []
    for i, s in enumerate(req.snippets, 1):
        body = (s.content or "").strip().replace("\n", " ")
        if len(body) > max_chars:
            body = body[:max_chars] + "…"
        lines.append(f"[{i}] {s.title} ({s.source_doc}::{s.node_id})\n    {body}")
    return "\n".join(lines)


def build_judge_user_message(
    *, rubric_name: str, req: ContentRequest, gen: ContentResult, allowed_values: list[str]
) -> str:
    """Assemble the user-role message for one judge call.

    The allowed indicator values are spelled out so the model cannot freelance
    outside the configured scale. The Pydantic ``response_format`` provides
    schema-level enforcement; this is the prose mirror.
    """
    rubric = RUBRICS.get(rubric_name) or (
        f"**{rubric_name}.** No detailed rubric was registered — judge on the "
        f"name alone."
    )
    return (
        f"Rubric:\n{rubric}\n\n"
        f"User query:\n  {req.refined_prompt}\n\n"
        f"Retrieved evidence:\n{_snippet_block(req)}\n\n"
        f"Generated draft:\n{gen.body}\n\n"
        f"Allowed indicator values: {allowed_values}\n"
        f"Return one of those values plus a one-sentence reason."
    )
