"""Read/write helpers for the article Markdown-with-frontmatter format.

Format:
    ---
    title: "..."
    source: "https://..."
    lang: "nl"
    published: 2026-01-19
    author:
      - "[[Name]]"
    description: "..."
    tag:
      - "[[Tag]]"
    ---
    <markdown body>

Frontmatter is read with PyYAML; written with a hand-rolled emitter that
matches `data/article/<lang>/` byte-for-byte (bare keys, double-quoted strings,
unquoted ISO dates, list items two-space-indented under their key).
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def parse_file(path: Path) -> tuple[dict[str, Any], str]:
    """Return (frontmatter dict, body str). Raises ValueError if no frontmatter."""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path} has no YAML frontmatter")
    fm = yaml.safe_load(m.group(1)) or {}
    body = text[m.end() :].lstrip("\n")
    return fm, body


def _yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_frontmatter(fm: dict[str, Any]) -> str:
    """Render frontmatter back to the canonical Obsidian-compatible YAML."""
    lines: list[str] = []
    for key, value in fm.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(
                    f"  - {_yaml_quote(item) if isinstance(item, str) else item}"
                )
        elif isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            lines.append(f"{key}: {value.isoformat()}")
        elif isinstance(value, str):
            lines.append(f"{key}: {_yaml_quote(value)}")
        elif value is None:
            lines.append(f"{key}: ")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def write_file(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_block = render_frontmatter(frontmatter)
    path.write_text(f"---\n{yaml_block}\n---\n{body}\n", encoding="utf-8")
