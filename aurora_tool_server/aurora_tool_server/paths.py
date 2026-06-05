"""Filesystem paths for the standalone package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
ASSETS_DIR = PROJECT_DIR / "assets"
PROFILES_DIR = ASSETS_DIR / "profiles"
RAG_DIR = ASSETS_DIR / "rag"
EVALUATION_DIR = ASSETS_DIR / "evaluation"
PROMPTS_DIR = ASSETS_DIR / "prompts"
