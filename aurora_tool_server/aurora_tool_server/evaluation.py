"""Deterministic KPI evaluation for phase 1."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from .paths import EVALUATION_DIR
from .schemas import Channel, ContentResult, EvaluationResult, KPIResult, Origin, Snippet


@lru_cache(maxsize=1)
def load_kpi_catalogue() -> dict[str, Any]:
    with (EVALUATION_DIR / "kpi_catalogue.json").open(encoding="utf-8") as f:
        return json.load(f)


def _catalogue_kpi(kpi_id: str, fallback_name: str, *, weight: str = "Medium") -> dict[str, Any]:
    for item in load_kpi_catalogue().get("kpis", []):
        if item.get("id") == kpi_id:
            return item
    return {
        "id": kpi_id,
        "name": fallback_name,
        "category": "Generic quality check",
        "weight": weight,
        "monitoring": "Mandatory",
    }


def _result(
    kpi_id: str,
    fallback_name: str,
    *,
    passed: bool,
    value: str,
    reason: str,
    weight: str = "Medium",
) -> KPIResult:
    item = _catalogue_kpi(kpi_id, fallback_name, weight=weight)
    return KPIResult(
        kpi_id=str(item.get("id") or kpi_id),
        name=str(item.get("name") or fallback_name),
        category=item.get("category"),
        weight=item.get("weight") or weight,
        monitoring=item.get("monitoring") or "Mandatory",
        value=value,
        reason=reason,
        tier=1,
        passed=passed,
        source="deterministic",
    )


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _maturity(results: list[KPIResult]) -> dict[str, str]:
    grouped: dict[str, list[KPIResult]] = {}
    for result in results:
        grouped.setdefault(result.category or "Uncategorised", []).append(result)
    out: dict[str, str] = {}
    for category, items in grouped.items():
        ratio = sum(1 for item in items if item.passed) / len(items)
        out[category] = "high" if ratio >= 0.8 else "medium" if ratio >= 0.5 else "low"
    return out


class _LLMKPIResult(BaseModel):
    kpi_id: str
    value: str
    passed: bool
    reason: str


class _LLMEvaluationResult(BaseModel):
    results: list[_LLMKPIResult] = Field(default_factory=list)
    reasoning: str = ""


_JUDGE_SYSTEM_PROMPT = """You are an ABN AMRO AURORA content quality judge.

Evaluate the generated content against the supplied KPI ids. Use the retrieved
snippets as the source-grounding reference. Return one result for each KPI id:
kpi_id, value, passed, reason. Keep reasons concise and audit-friendly.

Rules:
- Fail references when source-backed claims lack citations or citations do not
  correspond to retrieved snippets.
- Fail guardrails for unsupported absolutes, privacy/security risk, or unsafe
  claims.
- Judge content_quality_assessment on relevance, specificity, and usefulness.
- Judge reading_level on clarity and editorial readability.
"""


def _snippet_summary(snippets: list[Snippet]) -> str:
    lines: list[str] = []
    for index, snippet in enumerate(snippets[:8], start=1):
        excerpt = (snippet.content or "").strip().replace("\n", " ")[:500]
        lines.append(f"[{index}] {snippet.title} ({snippet.source_doc}::{snippet.node_id})\n{excerpt}")
    return "\n\n".join(lines) or "(no snippets)"


def _llm_evaluate(
    *,
    refined_prompt: str,
    content: ContentResult,
    snippets: list[Snippet],
    deterministic_results: list[KPIResult],
    api_key: str,
    model: str,
) -> _LLMEvaluationResult:
    from openai import OpenAI

    kpi_lines = "\n".join(
        f"- {result.kpi_id}: {result.name} (weight={result.weight}, "
        f"monitoring={result.monitoring})"
        for result in deterministic_results
    )
    citation_lines = "\n".join(
        f"- [{citation.index}] {citation.title} ({citation.source_doc}::{citation.node_id})"
        for citation in content.citations
    ) or "(no structured citations)"
    user_message = (
        f"Refined prompt:\n{refined_prompt}\n\n"
        f"Generated content:\n{content.body}\n\n"
        f"Structured citations:\n{citation_lines}\n\n"
        f"Retrieved snippets:\n{_snippet_summary(snippets)}\n\n"
        f"KPI ids to judge:\n{kpi_lines}\n"
    )
    client = OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=_LLMEvaluationResult,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM returned no parsed evaluation output")
    return parsed


def evaluate_draft(
    *,
    refined_prompt: str,
    content: ContentResult,
    snippets: list[Snippet],
    channel: Channel = "web",
    origin: Origin = "instant",
    api_key: str | None = None,
    model: str | None = None,
    strict_mode: bool = False,
) -> EvaluationResult:
    body = content.body.strip()
    sentences = _sentences(body)
    has_citation = bool(content.citations) or bool(re.search(r"\[\d+\]", body))
    has_source_grounding = bool(snippets)
    has_forbidden_claim = bool(
        re.search(r"\b(best|guaranteed|risk-free|most sustainable|always)\b", body, re.I)
    )

    results = [
        _result(
            "content_quality_assessment",
            "Content quality assessment",
            passed=len(body) >= 120 and len(sentences) >= 3,
            value="sufficient" if len(body) >= 120 and len(sentences) >= 3 else "insufficient",
            reason="Draft has enough substance for a phase-1 quality check.",
            weight="High",
        ),
        _result(
            "references",
            "References",
            passed=has_citation and has_source_grounding,
            value="present" if has_citation and has_source_grounding else "missing",
            reason="Generated content includes citation markers tied to retrieved snippets.",
            weight="Blocking",
        ),
        _result(
            "guardrails",
            "Guardrails",
            passed=not has_forbidden_claim,
            value="passed" if not has_forbidden_claim else "blocked_claim",
            reason="No unsupported superlative or absolute-risk claim detected.",
            weight="Blocking",
        ),
        _result(
            "reading_level",
            "Reading level",
            passed=(sum(len(s.split()) for s in sentences) / max(len(sentences), 1)) <= 28,
            value="plain" if sentences else "unknown",
            reason="Average sentence length is within a broad plain-language threshold.",
            weight="High",
        ),
    ]

    if strict_mode and not (api_key and model):
        results.append(
            _result(
                "llm_judge_required",
                "LLM judge required",
                passed=False,
                value="not_configured",
                reason="Strict mode requires a configured evaluation model.",
                weight="Blocking",
            )
        )

    source = "deterministic"
    model_used = None
    reasoning = "deterministic tier-1 phase-1 evaluation"
    if api_key and model:
        try:
            judged = _llm_evaluate(
                refined_prompt=refined_prompt,
                content=content,
                snippets=snippets,
                deterministic_results=results,
                api_key=api_key,
                model=model,
            )
            judged_by_id = {item.kpi_id: item for item in judged.results}
            llm_results: list[KPIResult] = []
            for base in results:
                item = judged_by_id.get(base.kpi_id)
                if item is None:
                    continue
                llm_results.append(
                    KPIResult(
                        kpi_id=base.kpi_id,
                        name=base.name,
                        category=base.category,
                        weight=base.weight,
                        monitoring=base.monitoring,
                        value=item.value,
                        reason=item.reason,
                        tier=2,
                        passed=item.passed,
                        source="llm",
                    )
                )
            if llm_results:
                results.extend(llm_results)
                source = "llm"
                model_used = model
                reasoning = judged.reasoning or "LLM judge evaluated tier-2 quality rubrics."
        except Exception as exc:
            reasoning = f"LLM evaluation failed; deterministic fallback: {exc}"
            if strict_mode:
                results.append(
                    _result(
                        "llm_judge_failed",
                        "LLM judge failed",
                        passed=False,
                        value="judge_error",
                        reason=str(exc),
                        weight="Blocking",
                    )
                )

    failed_blocking = sorted({
        result.kpi_id
        for result in results
        if result.weight == "Blocking" and result.monitoring == "Mandatory" and not result.passed
    })
    return EvaluationResult(
        passed=not failed_blocking,
        failed_blocking=failed_blocking,
        results=results,
        maturity_by_category=_maturity(results),
        dclp_steps_required=["editorial_review", "dclp_signoff"],
        channel=channel,
        origin=origin,
        model=model_used,
        source=source,
        reasoning=reasoning,
    )
