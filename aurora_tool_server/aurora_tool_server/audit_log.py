"""Audit event sink — structured stdout + append-only JSONL on disk.

Kept intentionally tiny: hosted demos expose stdout as the durable log surface
and disk is best-effort. Both writes are best-effort and never raise.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import PROJECT_DIR

AUDIT_LOG_PATH = Path(
    os.getenv("AURORA_AUDIT_LOG_PATH", str(PROJECT_DIR / "logs" / "audit.jsonl"))
)

_LOCK = threading.Lock()


def log_event(kind: str, payload: dict[str, Any]) -> None:
    """Emit a structured event to stdout and append to the JSONL log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        **payload,
    }
    line = json.dumps(record, default=str, ensure_ascii=False)

    print(f"audit_event {line}", file=sys.stdout, flush=True)

    try:
        with _LOCK:
            AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        pass
