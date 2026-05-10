"""Intent classification wrapper.

Reuses ``task_definition.IntentClassifier`` and its ``ClassificationResult``
schema. If both an API key and a model are configured, this wrapper makes
its own LLM call with the *configured* model (teammate's classifier
hardcodes the model). On any failure — or if either field is missing — it
delegates to teammate's deterministic fallback path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DEF_DIR = REPO_ROOT / "task_definition"
for _p in (str(REPO_ROOT), str(TASK_DEF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openai  # noqa: E402
from loguru import logger  # noqa: E402

from intent_classifier import ClassificationResult, IntentClassifier  # noqa: E402

# Mirrors the system prompt in task_definition/intent_classifier.py — kept in
# sync manually so we don't have to modify teammate's module to expose it.
_SYSTEM_PROMPT = """You are an intent classifier for ABN AMRO.
Classify the user's request into one of these task codes:
T1_DRAFT: Draft new content
T1_TRANSLATE: Translate existing content
T1_SEARCH: Search corpus for related articles
T2_COMPLIANCE: Quality & compliance check
T4_RENEWAL: Detect & renew aging articles

Assign a confidence score between 0.0 and 1.0.
Provide a short task reason.
Choose a role from: Insights Editorial, Chatbot (Anna), Mobile App (UX), Web / IB.
"""


def classify(
    prompt: str,
    api_key: str | None,
    model: str | None,
) -> tuple[str, str, float, str, str | None, str]:
    """Classify ``prompt``.

    Returns ``(role, task_code, confidence, reason, raw_json, source)`` where
    ``source`` is ``"llm"`` or ``"deterministic"``.
    """
    if api_key and model:
        try:
            logger.info(f"Frontend wrapper: classifying via LLM (model={model})...")
            client = openai.OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=ClassificationResult,
            )
            r = completion.choices[0].message.parsed
            return (
                r.role,
                r.task_code,
                r.confidence,
                r.task_reason,
                r.model_dump_json(indent=2),
                "llm",
            )
        except Exception as e:
            logger.warning(f"Frontend LLM classify failed ({e}); falling back to deterministic")

    role, task_code, confidence, reason, raw = IntentClassifier().classify(prompt, api_key=None)
    return role, task_code, confidence, reason, raw, "deterministic"
