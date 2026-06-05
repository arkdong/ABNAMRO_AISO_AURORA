"""Environment loading helpers for the standalone AURORA package."""

from __future__ import annotations

from dotenv import load_dotenv

from .paths import PROJECT_DIR


def load_project_env() -> None:
    """Load aurora_tool_server/.env without overriding shell-provided values."""

    load_dotenv(PROJECT_DIR / ".env", override=False)
