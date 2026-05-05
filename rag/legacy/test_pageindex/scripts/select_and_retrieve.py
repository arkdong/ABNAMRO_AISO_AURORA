#!/usr/bin/env python3
"""
Local POC for:
1) loading a small document registry,
2) selecting candidate documents from a prompt,
3) retrieving relevant PageIndex sections from generated tree JSON,
4) assembling the final context pack to send to an LLM.

This script does NOT generate PageIndex trees itself.
First run PageIndex on the source files and place the resulting *_structure.json files
in a folder such as ./pageindex_results or PageIndex/results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EDITORIAL_KEYWORDS = {
    "write", "writing", "rewrite", "rewriting", "refine", "refining", "review",
    "reviewing", "audit", "auditing", "align", "alignment", "draft", "article",
    "tone", "style", "voice", "headline", "intro", "introduction", "cta",
    "call", "action", "quality", "guideline", "guide", "checklist", "content",
    "insights", "schrijf", "herschrijf", "herformuleer", "review", "controleer",
    "check", "concept", "artikel", "toon", "stijl", "kop", "intro", "kwaliteit",
    "richtlijn", "handleiding", "inhoud"
}

STOPWORDS = {
    # EN
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "by",
    "from", "this", "that", "these", "those", "is", "are", "be", "as", "at",
    "it", "its", "into", "than", "then", "your", "you", "our", "we", "they",
    "their", "how", "what", "which", "when", "where", "why", "who", "about",
    "can", "should", "must", "will", "would", "may", "more", "most",
    # NL
    "de", "het", "een", "en", "of", "voor", "van", "in", "op", "met", "door",
    "uit", "dit", "dat", "deze", "die", "is", "zijn", "als", "bij", "om", "dan",
    "je", "jij", "jouw", "uw", "we", "wij", "ons", "onze", "hun", "hoe", "wat",
    "welke", "wanneer", "waar", "waarom", "wie", "kan", "kunnen", "moet",
    "moeten", "zal", "zullen", "meer", "meest"
}


def slug_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9à-ÿ]+", " ", text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]


def token_set(text: str) -> set[str]:
    return set(slug_tokens(text))


def is_editorial_query(query: str) -> bool:
    q = token_set(query)
    return bool(q & EDITORIAL_KEYWORDS)


def safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> Dict[str, Any]:
    return safe_read_json(path)


def find_result_path(results_dir: Path, result_file: str) -> Path:
    p = results_dir / result_file
    if p.exists():
        return p

    # Fallback: try a normalized search when file names differ slightly
    wanted = normalize_filename(result_file)
    candidates = list(results_dir.glob("*_structure.json"))
    for candidate in candidates:
        if normalize_filename(candidate.name) == wanted:
            return candidate

    raise FileNotFoundError(f"Could not find PageIndex result file '{result_file}' in {results_dir}")


def normalize_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def tree_root_nodes(tree_json: Any) -> List[Dict[str, Any]]:
    if isinstance(tree_json, list):
        return tree_json
    if isinstance(tree_json, dict):
        if "structure" in tree_json and isinstance(tree_json["structure"], list):
            return tree_json["structure"]
        if "result" in tree_json and isinstance(tree_json["result"], list):
            return tree_json["result"]
        if "nodes" in tree_json and isinstance(tree_json["nodes"], list):
            return tree_json["nodes"]
    return []


def flatten_nodes(nodes: Sequence[Dict[str, Any]], doc_meta: Dict[str, Any], parent_titles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parent_titles = parent_titles or []
    flat: List[Dict[str, Any]] = []

    for node in nodes:
        title = str(node.get("title", "")).strip()
        summary = str(
            node.get("summary")
            or node.get("node_summary")
            or ""
        ).strip()
        text = str(
            node.get("text")
            or node.get("node_text")
            or node.get("content")
            or ""
        ).strip()

        entry = {
            "doc_id": doc_meta["id"],
            "doc_title": doc_meta["title"],
            "doc_type": doc_meta["doc_type"],
            "sector": doc_meta.get("sector"),
            "node_id": str(node.get("node_id", "")),
            "title": title,
            "summary": summary,
            "text": text,
            "line_num": node.get("line_num"),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
            "path_titles": parent_titles + ([title] if title else []),
        }
        flat.append(entry)

        child_nodes = node.get("nodes") or []
        if isinstance(child_nodes, list) and child_nodes:
            flat.extend(flatten_nodes(child_nodes, doc_meta, parent_titles=entry["path_titles"]))

    return flat


def document_score(query: str, doc: Dict[str, Any], editorial: bool) -> float:
    q = token_set(query)
    joined = " ".join(
        [
            doc.get("title", ""),
            doc.get("description", ""),
            doc.get("sector", ""),
            doc.get("audience", ""),
            doc.get("doc_type", ""),
        ]
    )
    d = token_set(joined)
    overlap = len(q & d)

    score = float(overlap)

    if editorial and doc.get("doc_type") == "style_guide":
        score += 100.0

    if doc.get("doc_type") == "approved_example":
        score += 5.0

    # Topic shaping for current corpus
    if "technologie" in q or "technology" in q or "tmt" in q:
        if doc.get("sector") == "technologie":
            score += 6.0
    if "dienstverlening" in q or "services" in q or "zakelijke" in q:
        if doc.get("sector") == "zakelijke_dienstverlening":
            score += 6.0
    if "ai" in q or "agentic" in q or "kunstmatige" in q:
        if "ai" in d or "agentic" in d or "kunstmatige" in d:
            score += 4.0

    # Slight authority preference
    score += float(doc.get("authority_rank", 0)) / 100.0

    return score


def format_location(node: Dict[str, Any]) -> str:
    if node.get("line_num") is not None:
        return f"line {node['line_num']}"
    start_index = node.get("start_index")
    end_index = node.get("end_index")
    if start_index is not None and end_index is not None:
        if start_index == end_index:
            return f"page {start_index}"
        return f"pages {start_index}-{end_index}"
    return "location unknown"


def section_score(query: str, node: Dict[str, Any], editorial: bool) -> float:
    q = token_set(query)
    joined = " ".join(
        [
            node.get("title", ""),
            node.get("summary", ""),
            node.get("text", "")[:3000],  # keep scoring cheap
            " ".join(node.get("path_titles", [])),
            node.get("doc_title", ""),
        ]
    )
    n = token_set(joined)
    overlap = len(q & n)
    score = float(overlap)

    if node.get("doc_type") == "style_guide" and editorial:
        guide_boost_terms = {
            "structure", "heading", "headings", "subheading", "paragraph",
            "call", "action", "tone", "style", "wording", "message", "accuracy",
            "plain", "language", "short", "sentences", "active", "jargon",
            "british", "inclusive", "reader", "friendly"
        }
        score += 2.0 * len(guide_boost_terms & n)

    if node.get("doc_type") == "approved_example":
        example_boost_terms = {
            "introduction", "kansen", "groei", "ai", "agentic", "digitale",
            "soevereiniteit", "marketing", "cyber", "business", "sector"
        }
        score += 0.5 * len(example_boost_terms & n)

    # Slight preference for more specific nodes with some real text
    if node.get("text"):
        score += min(len(node["text"]) / 1000.0, 3.0)

    # Small preference for deeper nodes
    score += min(len(node.get("path_titles", [])) * 0.2, 1.0)

    return score


def summarize_text(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_document_selection_prompt(query: str, docs: Sequence[Dict[str, Any]]) -> str:
    docs_payload = [
        {
            "doc_id": d["id"],
            "doc_name": Path(d["path"]).name,
            "doc_type": d["doc_type"],
            "sector": d.get("sector"),
            "doc_description": d.get("description", ""),
        }
        for d in docs
    ]
    prompt = f"""
You are given a list of documents with IDs, file names, and descriptions.
Select the documents that are most useful for the user's editorial task.

Rules:
1. If the task is about writing, rewriting, reviewing, auditing, or aligning a draft with ABN AMRO quality, ALWAYS include the style guide.
2. Prefer approved examples from the same sector or topic as the draft.
3. For this small corpus, choose at most 4 documents total unless the task explicitly asks for broader comparison.
4. Return JSON only.

Output schema:
{{
  "thinking": "<brief reasoning>",
  "selected_doc_ids": ["doc_id_1", "doc_id_2"]
}}

User query:
{query}

Documents:
{json.dumps(docs_payload, ensure_ascii=False, indent=2)}
""".strip()
    return prompt


def build_section_retrieval_prompt(query: str, tree_json: Any) -> str:
    prompt = f"""
You are given a user query and the PageIndex tree structure of one selected document.
Find the sections that are most likely to contain useful context.

Rules:
1. Prefer sections that directly support the user task.
2. For the style guide, prioritize structure, tone of voice, wording, clarity, and accuracy rules.
3. For approved examples, prioritize introduction, framing, headline style, subheadings, and sections close to the topic.
4. Return JSON only.

Output schema:
{{
  "thinking": "<brief reasoning>",
  "selected_node_ids": ["0001", "0007"]
}}

User query:
{query}

Document tree:
{json.dumps(tree_json, ensure_ascii=False, indent=2)[:20000]}
""".strip()
    return prompt


def select_docs(query: str, docs: Sequence[Dict[str, Any]], max_docs: int) -> List[Dict[str, Any]]:
    editorial = is_editorial_query(query)
    scored = []
    for doc in docs:
        score = document_score(query, doc, editorial)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen: List[Dict[str, Any]] = []
    seen_ids = set()
    for score, doc in scored:
        if score <= 0 and chosen:
            continue
        if doc["id"] in seen_ids:
            continue
        enriched = dict(doc)
        enriched["_selection_score"] = round(score, 3)
        chosen.append(enriched)
        seen_ids.add(doc["id"])
        if len(chosen) >= max_docs:
            break

    # Enforce guide if editorial
    if editorial and not any(d["doc_type"] == "style_guide" for d in chosen):
        guide = next((d for d in docs if d["doc_type"] == "style_guide"), None)
        if guide:
            guide = dict(guide)
            guide["_selection_score"] = round(document_score(query, guide, editorial), 3)
            chosen = [guide] + chosen[: max_docs - 1]

    return chosen


def retrieve_sections(query: str, selected_docs: Sequence[Dict[str, Any]], results_dir: Path, max_sections_per_doc: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    editorial = is_editorial_query(query)
    all_sections: List[Dict[str, Any]] = []
    loaded_docs: List[Dict[str, Any]] = []

    for doc in selected_docs:
        result_path = find_result_path(results_dir, doc["result_file"])
        tree_json = safe_read_json(result_path)
        loaded_doc = dict(doc)
        loaded_doc["_result_path"] = str(result_path)
        loaded_doc["_selection_prompt"] = build_document_selection_prompt(query, selected_docs)
        loaded_doc["_section_prompt"] = build_section_retrieval_prompt(query, tree_json)
        loaded_docs.append(loaded_doc)

        flat_nodes = flatten_nodes(tree_root_nodes(tree_json), doc)
        node_scored = []
        for node in flat_nodes:
            score = section_score(query, node, editorial)
            node_scored.append((score, node))
        node_scored.sort(key=lambda x: x[0], reverse=True)

        kept = 0
        for score, node in node_scored:
            if kept >= max_sections_per_doc:
                break
            if score <= 0:
                continue
            item = dict(node)
            item["score"] = round(score, 3)
            item["location"] = format_location(node)
            item["excerpt"] = summarize_text(node.get("text") or node.get("summary") or "")
            all_sections.append(item)
            kept += 1

    all_sections.sort(key=lambda x: x["score"], reverse=True)
    return loaded_docs, all_sections


def assemble_context(query: str, selected_docs: Sequence[Dict[str, Any]], sections: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"USER TASK: {query}")
    lines.append("")
    lines.append("CONTEXT PACK")
    lines.append("=" * 12)

    docs_by_id = {d["id"]: d for d in selected_docs}

    # Normative docs first
    ordered_sections = sorted(
        sections,
        key=lambda s: (
            0 if docs_by_id.get(s["doc_id"], {}).get("doc_type") == "style_guide" else 1,
            -float(s["score"]),
            s["doc_title"].lower(),
        )
    )

    for i, sec in enumerate(ordered_sections, start=1):
        doc = docs_by_id.get(sec["doc_id"], {})
        lines.append(f"{i}. {sec['doc_title']} [{doc.get('doc_type', 'unknown')}]")
        lines.append(f"   section: {' > '.join(sec.get('path_titles', [])) or sec.get('title', '(untitled)')}")
        lines.append(f"   ref: {sec['location']} | node_id={sec.get('node_id', '')} | score={sec['score']}")
        if doc.get("description"):
            lines.append(f"   doc description: {doc['description']}")
        if sec.get("excerpt"):
            lines.append(f"   excerpt: {sec['excerpt']}")
        else:
            lines.append("   excerpt: [no node text available; rely on summary]")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select documents and retrieve relevant PageIndex sections for a local ABN AMRO Insights POC.")
    parser.add_argument("--manifest", required=True, help="Path to corpus_manifest.json")
    parser.add_argument("--pageindex-results", required=True, help="Directory with PageIndex *_structure.json files")
    parser.add_argument("--query", required=True, help="User prompt or editorial task")
    parser.add_argument("--max-docs", type=int, default=4, help="Maximum number of documents to select")
    parser.add_argument("--max-sections-per-doc", type=int, default=3, help="Maximum sections to keep per selected document")
    parser.add_argument("--output-json", help="Optional path to save the full result as JSON")
    parser.add_argument("--print-prompts", action="store_true", help="Print the prompt templates filled with your data")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    docs = manifest["documents"]

    selected_docs = select_docs(args.query, docs, max_docs=args.max_docs)
    loaded_docs, sections = retrieve_sections(
        args.query,
        selected_docs,
        results_dir=Path(args.pageindex_results),
        max_sections_per_doc=args.max_sections_per_doc,
    )

    context = assemble_context(args.query, loaded_docs, sections)

    result = {
        "query": args.query,
        "selected_documents": [
            {
                "id": d["id"],
                "title": d["title"],
                "doc_type": d["doc_type"],
                "sector": d.get("sector"),
                "selection_score": d.get("_selection_score"),
                "result_file": d["result_file"],
                "result_path": d.get("_result_path"),
            }
            for d in loaded_docs
        ],
        "selected_sections": sections,
        "assembled_context": context,
    }

    print("\n=== SELECTED DOCUMENTS ===")
    for d in result["selected_documents"]:
        print(f"- {d['title']} [{d['doc_type']}] | score={d['selection_score']}")

    print("\n=== SELECTED SECTIONS ===")
    for s in sections:
        section_path = " > ".join(s.get("path_titles", [])) or s.get("title", "(untitled)")
        print(f"- {s['doc_title']} | {section_path} | {s['location']} | score={s['score']}")

    print("\n=== ASSEMBLED CONTEXT ===")
    print(context)

    if args.print_prompts and loaded_docs:
        print("\n=== DOCUMENT SELECTION PROMPT ===")
        print(build_document_selection_prompt(args.query, docs))
        print("\n=== SECTION RETRIEVAL PROMPT (FIRST SELECTED DOC) ===")
        tree_json = safe_read_json(Path(loaded_docs[0]["_result_path"]))
        print(build_section_retrieval_prompt(args.query, tree_json))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved result JSON to {out_path}")


if __name__ == "__main__":
    main()
