"""Load cached PageIndex JSONs into a normalised form.

``rag/corpus/corpus_en_structure.json`` wraps the tree in a dict; the writing
guide tree is a bare list at top level. Both get normalised to a flat
:class:`CorpusDoc` whose ``nodes`` are the top-level tree nodes (each with
``title``, ``node_id``, ``text``, optional ``line_num`` / ``page_index``,
``prefix_summary`` / ``summary``, ``nodes`` for children).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "rag" / "corpus"

CORPUS_FILES: dict[str, Path] = {
    "corpus_en": CORPUS_DIR / "corpus_en_structure.json",
    "writing_guide": CORPUS_DIR / "writing_guide_tree.json",
}


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    doc_name: str
    nodes: tuple[dict, ...]


def _normalize(doc_id: str, raw: Any) -> CorpusDoc:
    if isinstance(raw, dict):
        return CorpusDoc(
            doc_id=doc_id,
            doc_name=raw.get("doc_name", doc_id),
            nodes=tuple(raw.get("structure", [])),
        )
    if isinstance(raw, list):
        return CorpusDoc(doc_id=doc_id, doc_name=doc_id, nodes=tuple(raw))
    raise ValueError(f"unknown structure for {doc_id}")


def load_corpora() -> dict[str, CorpusDoc]:
    """Read every cached corpus JSON we know about. Missing files are skipped silently."""
    out: dict[str, CorpusDoc] = {}
    for doc_id, path in CORPUS_FILES.items():
        if not path.exists():
            continue
        with path.open() as f:
            raw = json.load(f)
        out[doc_id] = _normalize(doc_id, raw)
    return out


def walk_nodes(nodes: Iterable[dict]) -> Iterator[dict]:
    """Yield every node in the tree, depth-first."""
    for n in nodes:
        yield n
        children = n.get("nodes") or ()
        if children:
            yield from walk_nodes(children)
