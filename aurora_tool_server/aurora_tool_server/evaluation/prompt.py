"""Prompts for Tier 2 LLM judges.

One shared system message, one anchor per rubric. The user message is
assembled by :func:`build_judge_user_message` from the request / generation
/ retrieved snippets.
"""

from __future__ import annotations

from typing import Optional

from ..schemas import ContentResult
from .catalogue import KPI
from .inputs import EvalRequest

JUDGE_SYSTEM_PROMPT = """You are a content-quality auditor at ABN AMRO.

You will receive:
- A *user query* (what the customer asked for).
- A list of *retrieved evidence snippets* (the corpus context the writer had).
- A *generated content* draft.
- A *rubric* describing exactly one quality dimension to score.
- An *indicator scale* — the only allowed values for your score.

Your job: score the draft on this rubric only for a generation-time review.
Return:
- ``value``: one value from the indicator scale, verbatim.
- ``reason``: one short sentence (max 25 words) justifying the score.

Rules:
- Be pragmatic and evidence-based. Borderline cases default to the less severe value.
- Reserve severe values for material issues a human editor must fix before use.
- Do not invent claims about the draft that are not in it.
- Do not penalize harmless omissions, concise drafting, or style differences unless the rubric asks for them.
- The scale's allowed values are listed in the rubric; pick exactly one.
- If you cannot judge (e.g. draft is empty), return ``"unknown"``.
"""


RUBRICS: dict[str, str] = {
    "factuality": (
        "**Factuality.** Count material factual errors in the draft. "
        "A factual error is a substantive claim that contradicts the "
        "retrieved evidence or introduces unsupported specifics such as "
        "made-up numbers, dates, attributions, or guarantees. Minor wording, "
        "common background context, and harmless omissions do not count."
    ),
    "truthfullness": (
        "**Truthfulness.** Identify material deviations from bank standards for "
        "truthful substance (e.g. using 'advice' for execution-only services, "
        "claiming guarantees the bank does not actually offer, dramatising "
        "risk/return). Ignore harmless wording differences."
    ),
    "relevancy": (
        "**Relevancy.** Does the draft address the user's actual query? "
        "Use off-topic only when the draft largely fails to answer what the "
        "customer asked for. Do not judge writing polish here."
    ),
    "groundedness": (
        "**Groundedness.** What share of the draft's substantive claims can "
        "be traced back to the retrieved snippets? Full = every material "
        "claim anchored; reasonable = most; limited = a few; none = none."
    ),
    "privacy_security": (
        "**Privacy & security.** Identify concrete deviations from bank "
        "policy: exposing personal data, presenting non-AAB external domains "
        "as bank action links, recommending insecure flows, mishandling "
        "BSN/IBAN, etc. Use severe values only for material policy risks."
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
    "reading_level": (
        "**Reading level (CEFR).** Judge the CEFR language level of the "
        "draft (A1 = simplest … C2 = most complex). Consider sentence "
        "length and structure, word frequency/difficulty, and abstraction. "
        "The bank's norm for customer content is B1 or simpler."
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


def _snippet_block(req: EvalRequest, max_chars: int = 600) -> str:
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
    *,
    rubric_name: str,
    req: EvalRequest,
    gen: ContentResult,
    allowed_values: list[str],
    kpi: Optional[KPI] = None,
) -> str:
    """Assemble the user-role message for one judge call.

    The allowed indicator values are spelled out so the model cannot freelance
    outside the configured scale. The Pydantic ``response_format`` provides
    schema-level enforcement; this is the prose mirror. When the KPI is
    supplied, the workbook norm — including the channel deviant norm (e.g.
    "no attachments in chat") — anchors the judgement.
    """
    rubric = RUBRICS.get(rubric_name) or (
        f"**{rubric_name}.** No detailed rubric was registered — judge on the "
        f"name alone."
    )
    norm_block = ""
    if kpi is not None:
        norm = kpi.norm_for(req.channel)
        if norm:
            norm_block = f"Bank norm for this KPI ({req.channel} channel):\n  {norm}\n\n"
    return (
        f"Rubric:\n{rubric}\n\n"
        f"{norm_block}"
        f"User query:\n  {req.refined_prompt}\n\n"
        f"Retrieved evidence:\n{_snippet_block(req)}\n\n"
        f"Generated draft:\n{gen.body}\n\n"
        f"Allowed indicator values: {allowed_values}\n"
        f"Return one of those values plus a one-sentence reason."
    )
