"""task_definition package.

Submodules (``intent_classifier``, ``main``, ``corpus_manager`` …) use bare
imports like ``from config import TASK_HINTS`` that assume this directory
is on ``sys.path``. We register it here so callers who import via the
package path (``from task_definition.intent_classifier import …``) don't
have to know that trick.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
