"""Content generation — backend module (stage 5).

Final stage of the pipeline: takes the locked-in refined prompt, the
selected profile bundle, and the retrieved snippets, and asks an LLM to
produce the actual content the user requested.

Public surface:

- :class:`ContentRequest`, :class:`ContentResult`, :class:`Citation` —
  data shapes.
- :func:`generate_content` — single entry point. Falls back to a
  deterministic stub when no API key / model is configured.
"""

from __future__ import annotations

from backend.content_generation.service import generate_content
from backend.content_generation.types import (
    Citation,
    ContentRequest,
    ContentResult,
)

__all__ = [
    "Citation",
    "ContentRequest",
    "ContentResult",
    "generate_content",
]
