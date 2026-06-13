"""Workshop telemetry sink — structured stdout + append-only JSONL on disk.

Kept intentionally tiny: workshops run on Railway where stdout is the durable
log surface and disk is best-effort. Both writes are best-effort and never raise.
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

WORKSHOP_LOG_PATH = Path(
    os.getenv("AURORA_WORKSHOP_LOG_PATH", str(PROJECT_DIR / "assets" / "workshop" / "workshop.jsonl"))
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

    print(f"workshop_event {line}", file=sys.stdout, flush=True)

    try:
        with _LOCK:
            WORKSHOP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with WORKSHOP_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        pass
