"""Profile selection — backend module.

Maps an :class:`backend.intent.IntentResult` to a
:class:`profiles.ProfileBundle` via :func:`select`.
"""

from __future__ import annotations

from backend.profile_selection.selector import select

__all__ = ["select"]
