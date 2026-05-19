"""Intent classification — backend module.

Public surface:

- :class:`IntentResult` — rich classification schema (task_codes, sector,
  topic_keywords, language, plus role/confidence/task_reason).
- :func:`classify_full` — preferred entry point; returns
  ``(IntentResult, source)`` where ``source`` is ``"llm"`` or
  ``"deterministic"``.
- :func:`classify` — legacy 6-tuple shim for callers that haven't migrated.
"""

from __future__ import annotations

from backend.intent.classifier import (
    IntentResult,
    classify,
    classify_full,
)

__all__ = ["IntentResult", "classify", "classify_full"]
