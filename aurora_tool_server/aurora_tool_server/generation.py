"""Draft generation with deterministic fallback."""

from __future__ import annotations

import json

from .schemas import Citation, ContentResult, IntentResult, ProfileBundleResult, Snippet


def _language_label(intent: IntentResult) -> str:
    return {
        "nl": "Dutch",
        "en": "English",
        "both": "Dutch and English",
    }.get(intent.language or "en", "English")


def _language_rules(intent: IntentResult) -> str:
    if intent.language == "nl":
        return (
            "Write in Dutch. Follow the Nederlandse ABN AMRO Schrijfwijzer: "
            "use formal u-form, B1/plain language, active sentences, clear "
            "headings, and ABN AMRO Insights tone. Do not switch to English "
            "except for proper nouns, cited source titles, or unavoidable "
            "technical terms."
        )
    if intent.language == "both":
        return (
            "Return both Dutch and English versions in clearly labelled Markdown "
            "sections. The Dutch version must follow the Schrijfwijzer and use "
            "formal u-form; the English version must use British English."
        )
    return (
        "Write in English using British English, ABN AMRO Insights tone, "
        "plain language, active sentences, and clear headings."
    )


def _stub_body(
    refined_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundleResult,
    snippets: list[Snippet],
) -> str:
    if intent.language == "nl":
        title = "AURORA onderbouwde concepttekst"
        if intent.topic_keywords:
            title = f"AURORA onderbouwde concepttekst: {intent.topic_keywords[0]}"
        profile_names = ", ".join(p.name for p in profiles.all_profiles) or "geen gekoppeld profiel"
        source_lines = []
        for index, snippet in enumerate(snippets[:5], start=1):
            source_lines.append(
                f"- [{index}] {snippet.title} ({snippet.source_doc}::{snippet.node_id})"
            )
        source_block = "\n".join(source_lines) or "- Er zijn geen bronfragmenten opgehaald."
        return (
            f"# {title}\n\n"
            "Deze deterministische concepttekst houdt de AURORA-stroom intact "
            "wanneer er geen contentgeneratiemodel is geconfigureerd. In productie "
            "moet de LLM-generatie dit vervangen, maar de tekst bewaart wel de "
            "verfijnde instructie, actieve profielen en bronbasis voor auditbaarheid.\n\n"
            f"**Verfijnde instructie**\n\n{refined_prompt}\n\n"
            f"**Actieve profielen**\n\n{profile_names}\n\n"
            "**Gebruikte broncontext**\n\n"
            f"{source_block}\n\n"
            "Een definitieve redactionele tekst moet de opgehaalde bronnen gebruiken, "
            "de ABN AMRO Schrijfwijzer volgen, onbewezen claims vermijden en "
            "citaten gekoppeld houden aan brononderbouwde uitspraken."
        )

    title = "AURORA grounded draft"
    if intent.topic_keywords:
        title = f"AURORA grounded draft: {intent.topic_keywords[0]}"
    profile_names = ", ".join(p.name for p in profiles.all_profiles) or "no matched profile"
    source_lines = []
    for index, snippet in enumerate(snippets[:5], start=1):
        source_lines.append(
            f"- [{index}] {snippet.title} ({snippet.source_doc}::{snippet.node_id})"
        )
    source_block = "\n".join(source_lines) or "- No source snippets were retrieved."
    return (
        f"# {title}\n\n"
        "This phase-1 deterministic draft preserves the AURORA flow when no "
        "content-generation model is configured. It should be replaced by the "
        "LLM generation path in production, but it still carries the refined "
        "instruction, active profiles, and source grounding for auditability.\n\n"
        f"**Refined instruction**\n\n{refined_prompt}\n\n"
        f"**Active profiles**\n\n{profile_names}\n\n"
        "**Grounded context used**\n\n"
        f"{source_block}\n\n"
        "A final editorial draft should use the retrieved evidence above, follow "
        "the ABN AMRO writing guide, avoid unsupported claims, and keep citations "
        "attached to source-backed statements."
    )


def _citations(snippets: list[Snippet]) -> list[Citation]:
    return [
        Citation(
            index=index,
            source_doc=snippet.source_doc,
            node_id=snippet.node_id,
            title=snippet.title,
        )
        for index, snippet in enumerate(snippets[:5], start=1)
    ]


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    return json.loads(content)


def _body_from_parsed(parsed: dict) -> str:
    """Extract draft markdown from common LLM JSON key variants."""
    for key in ("body", "body_markdown", "markdown", "content", "draft", "text"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def generate_draft(
    *,
    refined_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundleResult,
    snippets: list[Snippet],
    api_key: str | None = None,
    model: str | None = None,
) -> ContentResult:
    if api_key and model:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            snippets_text = "\n".join(
                f"[{i}] {s.title} ({s.source_doc}::{s.node_id})\n{s.content[:1200]}"
                for i, s in enumerate(snippets[:8], start=1)
            )
            language_rules = _language_rules(intent)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write grounded ABN AMRO editorial content using only "
                            "approved retrieved context. Preserve citations and avoid "
                            "unsupported claims. Return JSON with body markdown, "
                            "reasoning, and citation_indices.\n\n"
                            f"Output language: {_language_label(intent)}.\n"
                            f"Language rules: {language_rules}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Instruction:\n{refined_prompt}\n\n"
                            f"Output language: {_language_label(intent)}\n"
                            f"Language rules:\n{language_rules}\n\n"
                            f"Sources:\n{snippets_text}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            parsed = _parse_json(completion.choices[0].message.content or "{}")
            body = _body_from_parsed(parsed)
            if not body.strip():
                raise ValueError("LLM returned empty body")
            citation_indices = parsed.get("citation_indices") or []
            valid = [snippets[i - 1] for i in citation_indices if isinstance(i, int) and 1 <= i <= len(snippets)]
            return ContentResult(
                body=body,
                citations=_citations(valid),
                model=model,
                source="llm",
                reasoning=str(parsed.get("reasoning") or "generated by configured LLM"),
            )
        except Exception as exc:
            return ContentResult(
                body=_stub_body(refined_prompt, intent, profiles, snippets),
                citations=_citations(snippets),
                source="deterministic",
                reasoning=f"stub fallback after LLM error: {exc}",
            )

    return ContentResult(
        body=_stub_body(refined_prompt, intent, profiles, snippets),
        citations=_citations(snippets),
        source="deterministic",
        reasoning="stub fallback because no content-generation API key/model is configured",
    )
