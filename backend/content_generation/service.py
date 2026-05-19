"""Content generation entry point (Stage 5).

Mirrors :mod:`backend.prompt_refinement.generator`: structured-output OpenAI
call via ``client.beta.chat.completions.parse``, with a deterministic stub
when no ``api_key`` / ``model`` is configured so the UI stays usable in dev.
"""

from __future__ import annotations

from typing import Optional

import openai
from loguru import logger

from backend.content_generation.prompt import SYSTEM_PROMPT, build_user_message
from backend.content_generation.types import Citation, ContentRequest, ContentResult


def _stub(req: ContentRequest, *, reason: str) -> ContentResult:
    """Deterministic placeholder used when the LLM path is unavailable."""
    snippet_list = "\n".join(
        f"- [{i + 1}] {s.title} ({s.source_doc}::{s.node_id})"
        for i, s in enumerate(req.snippets[:5])
    ) or "(no snippets)"
    body = (
        "*(Stub content — LLM not configured. Set "
        "`OPENAI_API_KEY_CONTENT_GENERATION` and a model on the Settings "
        "page to enable real generation.)*\n\n"
        f"**Refined prompt**\n\n> {req.refined_prompt}\n\n"
        f"**Snippets available**\n\n{snippet_list}\n"
    )
    return ContentResult(
        body=body,
        citations=[],
        model=None,
        source="deterministic",
        reasoning=reason,
    )


def _filter_valid_citations(
    citations: list[Citation], req: ContentRequest
) -> list[Citation]:
    """Drop citations whose index is out of range for the snippet list.

    Belt-and-suspenders for hallucinated indices. We rewrite the
    ``source_doc``/``node_id``/``title`` from the snippet list rather than
    trust the model to echo them back exactly, so the UI always renders
    the real source.
    """
    valid: list[Citation] = []
    n = len(req.snippets)
    for c in citations:
        if c.index < 1 or c.index > n:
            logger.warning(
                f"Content generation: dropping citation index {c.index} "
                f"(only {n} snippets available)"
            )
            continue
        s = req.snippets[c.index - 1]
        valid.append(
            Citation(
                index=c.index,
                source_doc=s.source_doc,
                node_id=s.node_id,
                title=s.title,
            )
        )
    return valid


def generate_content(
    req: ContentRequest,
    *,
    api_key: Optional[str],
    model: Optional[str],
) -> ContentResult:
    """Produce the final content for the refined prompt.

    Returns the deterministic stub when ``api_key`` or ``model`` is unset,
    or when the LLM call raises — never propagates errors to the UI.
    """
    if not api_key or not model:
        logger.info("Content generation: stub path (no api_key/model)")
        return _stub(req, reason="stub — no LLM configured")

    user_msg = build_user_message(req)
    try:
        client = openai.OpenAI(api_key=api_key)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=ContentResult,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no parsed output")
        parsed = parsed.model_copy(
            update={
                "model": model,
                "source": "llm",
                "citations": _filter_valid_citations(parsed.citations, req),
            }
        )
        logger.info(
            f"Content generation: LLM path "
            f"(model={model}, body_chars={len(parsed.body)}, "
            f"citations={len(parsed.citations)})"
        )
        return parsed
    except Exception as e:
        logger.warning(f"Content generation: LLM call failed ({e}); using stub")
        return _stub(req, reason=f"stub fallback — LLM error: {e}")
