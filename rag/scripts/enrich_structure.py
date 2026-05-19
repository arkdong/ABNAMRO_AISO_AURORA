"""Join the article manifest into the PageIndex tree.

PageIndex's ``md_to_tree`` only sees the markdown body, so the top-level
(``#``) nodes in ``corpus_en_structure.json`` end up with whatever
``prefix_summary`` the LLM produced from the body's opening sentences plus
nothing about tags, publication date, or source. The article frontmatter has
all of that — this script joins the two by title, in place.

Mutations on each matched top-level node:

- ``prefix_summary`` is overwritten with the frontmatter ``description``
  (the cleanest article-level summary available).
- ``tags``, ``published``, ``source``, ``slug`` are added as extra fields.

Unmatched manifest entries / tree nodes are reported but not fatal — the
``pageindex_provider`` reader is tolerant of missing optional fields.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_PATH = REPO_ROOT / "rag" / "corpus" / "corpus_en_structure.json"
MANIFEST_PATH = REPO_ROOT / "rag" / "corpus" / "corpus_en_manifest.json"


def _normalise_title(t: str) -> str:
    """Collapse whitespace and case so frontmatter titles match tree titles
    even if smart-quote / punctuation differences sneak in."""
    return " ".join((t or "").split()).strip().lower()


def enrich(structure_path: Path = STRUCTURE_PATH, manifest_path: Path = MANIFEST_PATH) -> None:
    structure_doc = json.loads(structure_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_title = {_normalise_title(e["title"]): e for e in manifest}
    used: set[str] = set()
    matched = 0
    nodes = structure_doc.get("structure", [])

    for node in nodes:
        key = _normalise_title(node.get("title", ""))
        entry = by_title.get(key)
        if entry is None:
            continue
        used.add(key)
        matched += 1

        if entry.get("description"):
            node["prefix_summary"] = entry["description"]
        if entry.get("tags"):
            node["tags"] = entry["tags"]
        if entry.get("published"):
            node["published"] = entry["published"]
        if entry.get("source"):
            node["source"] = entry["source"]
        if entry.get("slug"):
            node["slug"] = entry["slug"]

    unmatched_nodes = [n["title"] for n in nodes if _normalise_title(n.get("title", "")) not in used]
    unmatched_manifest = [e["title"] for e in manifest if _normalise_title(e["title"]) not in used]

    structure_path.write_text(
        json.dumps(structure_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Enriched {matched}/{len(nodes)} top-level nodes from {len(manifest)} manifest entries")
    if unmatched_nodes:
        print(f"  tree nodes with no manifest match ({len(unmatched_nodes)}):")
        for t in unmatched_nodes:
            print(f"    - {t}")
    if unmatched_manifest:
        print(f"  manifest entries with no tree match ({len(unmatched_manifest)}):")
        for t in unmatched_manifest:
            print(f"    - {t}")


if __name__ == "__main__":
    enrich()
