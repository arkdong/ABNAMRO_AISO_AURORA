"""PageIndex-style retrieval with optional LLM ranking over copied local assets."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

from pydantic import BaseModel, Field

from .paths import RAG_DIR
from .schemas import (
    IntentResult,
    ProfileBundleResult,
    RetrievalQuery,
    RetrievalResult,
    Snippet,
)

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "about",
    "what",
    "when",
    "where",
    "which",
    "your",
    "our",
    "are",
    "was",
    "were",
    "het",
    "een",
    "voor",
    "met",
    "van",
    "zijn",
    "dat",
    "deze",
    "naar",
}

_ARTICLE_CORPORA = {
    "en": ("corpus_en",),
    "nl": ("corpus_nl",),
    "both": ("corpus_nl", "corpus_en"),
}

_WRITING_GUIDES = {
    "en": ("writing_guide",),
    "nl": ("schrijfwijzer",),
    "both": ("schrijfwijzer", "writing_guide"),
}


def _tokens(text: str, *, min_len: int = 3) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) >= min_len and token.lower() not in _STOPWORDS
    }


def _token_counts(text: str, *, min_len: int = 3) -> Counter[str]:
    return Counter(
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) >= min_len and token.lower() not in _STOPWORDS
    )


def _walk(nodes: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for node in nodes:
        yield node
        yield from _walk(node.get("nodes") or [])


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _walk_with_article_metadata(
    nodes: Iterable[dict[str, Any]],
    *,
    article_title: str | None = None,
    source_url: str | None = None,
) -> Iterable[tuple[dict[str, Any], str | None, str | None]]:
    for node in nodes:
        current_article_title = article_title or _clean_str(node.get("title"))
        current_source_url = _clean_str(node.get("source")) or source_url
        yield node, current_article_title, current_source_url
        yield from _walk_with_article_metadata(
            node.get("nodes") or [],
            article_title=current_article_title,
            source_url=current_source_url,
        )


def _load_json(path_name: str) -> Any:
    with (RAG_DIR / path_name).open(encoding="utf-8") as f:
        return json.load(f)


def _load_optional_json(path_name: str) -> Any | None:
    path = RAG_DIR / path_name
    if not path.is_file():
        return None
    return _load_json(path_name)


@lru_cache(maxsize=1)
def load_corpora() -> dict[str, tuple[dict[str, Any], ...]]:
    out: dict[str, tuple[dict[str, Any], ...]] = {}
    corpus_en = _load_optional_json("corpus_en_structure.json")
    if isinstance(corpus_en, dict):
        out["corpus_en"] = tuple(corpus_en.get("structure") or [])
    corpus_nl = _load_optional_json("corpus_nl_structure.json")
    if isinstance(corpus_nl, dict):
        out["corpus_nl"] = tuple(corpus_nl.get("structure") or [])
    writing_guide = _load_optional_json("writing_guide_tree.json")
    if isinstance(writing_guide, list):
        out["writing_guide"] = tuple(writing_guide)
    schrijfwijzer = _load_optional_json("schrijfwijzer_tree.json")
    if isinstance(schrijfwijzer, list):
        out["schrijfwijzer"] = tuple(schrijfwijzer)
    return out


@lru_cache(maxsize=1)
def load_article_metadata_index() -> dict[str, dict[str, dict[str, str | None]]]:
    out: dict[str, dict[str, dict[str, str | None]]] = {}
    for corpus_id, roots in load_corpora().items():
        by_node_id: dict[str, dict[str, str | None]] = {}
        for node, article_title, source_url in _walk_with_article_metadata(roots):
            node_id = _clean_str(node.get("node_id"))
            if node_id:
                by_node_id[node_id] = {
                    "article_title": article_title,
                    "source_url": source_url,
                }
        out[corpus_id] = by_node_id
    return out


def build_query(
    user_prompt: str,
    intent: IntentResult,
    profiles: ProfileBundleResult,
    *,
    k: int = 5,
    retrieval_backend: str = "pageindex",
) -> RetrievalQuery:
    return RetrievalQuery(
        user_prompt=user_prompt,
        task_codes=list(intent.task_codes),
        sector=intent.sector,
        topic_keywords=list(intent.topic_keywords),
        language=intent.language,
        workflow_profile_ids=[profile.id for profile in profiles.workflow],
        expert_profile_ids=[profile.id for profile in profiles.domain_expert],
        k=k,
        retrieval_backend=retrieval_backend,  # type: ignore[arg-type]
    )


def _route_corpora(query: RetrievalQuery, available: set[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    lang = query.language or "en"
    article_corpora = _ARTICLE_CORPORA.get(lang, _ARTICLE_CORPORA["en"])
    writing_guides = _WRITING_GUIDES.get(lang, _WRITING_GUIDES["en"])

    def add(corpus_ids: Iterable[str]) -> None:
        for corpus_id in corpus_ids:
            if corpus_id in available and corpus_id not in seen:
                seen.add(corpus_id)
                out.append(corpus_id)

    for code in query.task_codes or ["T1_DRAFT"]:
        if code == "T1_SEARCH":
            add(article_corpora)
        elif code == "T2_COMPLIANCE":
            add(writing_guides)
            add(article_corpora)
        elif code == "T1_TRANSLATE":
            add(article_corpora)
            add(writing_guides)
            if lang != "both":
                add(_ARTICLE_CORPORA["both"])
                add(_WRITING_GUIDES["both"])
        else:
            add(article_corpora)
            add(writing_guides)
    return out or sorted(available)


def _query_terms(query: RetrievalQuery) -> set[str]:
    text = " ".join(
        [
            query.user_prompt,
            query.sector or "",
            " ".join(query.topic_keywords),
            " ".join(query.workflow_profile_ids),
            " ".join(query.expert_profile_ids),
        ]
    )
    return _tokens(text) | _tokens(" ".join(query.topic_keywords), min_len=2)


def _query_text(query: RetrievalQuery) -> str:
    return " ".join(
        [
            query.user_prompt,
            query.sector or "",
            " ".join(query.topic_keywords),
            " ".join(query.workflow_profile_ids),
            " ".join(query.expert_profile_ids),
        ]
    )


def _tag_terms(query: RetrievalQuery) -> list[str]:
    return [kw.lower() for kw in query.topic_keywords if kw]


def _tag_matches(node: dict[str, Any], terms: list[str]) -> bool:
    tags = [str(tag).lower() for tag in (node.get("tags") or [])]
    if not tags or not terms:
        return False
    for term in terms:
        for tag in tags:
            if term in tag or tag in term:
                return True
    return False


def _filter_by_tags(
    nodes: tuple[dict[str, Any], ...],
    query: RetrievalQuery,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    terms = _tag_terms(query)
    if not terms:
        return nodes, False
    matched = tuple(node for node in nodes if _tag_matches(node, terms))
    if 0 < len(matched) < len(nodes):
        return matched, True
    return nodes, False


def _score_node(node: dict[str, Any], query_terms: set[str]) -> tuple[int, list[str]]:
    title = node.get("title") or ""
    summary = f"{node.get('prefix_summary') or ''} {node.get('summary') or ''}"
    tags = " ".join(str(tag) for tag in (node.get("tags") or []))
    title_tokens = _tokens(title)
    summary_tokens = _tokens(summary)
    tag_tokens = _tokens(tags, min_len=2)
    matched = sorted((title_tokens | summary_tokens | tag_tokens) & query_terms)
    score = (
        4 * len(title_tokens & query_terms)
        + 2 * len(tag_tokens & query_terms)
        + len(summary_tokens & query_terms)
    )
    return score, matched


def _snippet_from_node(
    *,
    corpus_id: str,
    node: dict[str, Any],
    score: float,
    reason: str,
    article_title: str | None,
    source_url: str | None,
) -> Snippet:
    return Snippet(
        source_doc=corpus_id,
        node_id=str(node.get("node_id") or ""),
        title=str(node.get("title") or ""),
        article_title=article_title or _clean_str(node.get("title")),
        content=str(node.get("text") or ""),
        line_num=node.get("line_num") or node.get("page_index"),
        score=score,
        reason=reason,
        source_url=_clean_str(node.get("source")) or source_url,
    )


def _deterministic_pick(query: RetrievalQuery) -> tuple[list[Snippet], list[str]]:
    corpora = load_corpora()
    routed = _route_corpora(query, set(corpora))
    query_terms = _query_terms(query)
    candidates: list[tuple[int, str, dict[str, Any], list[str], str | None, str | None]] = []

    for corpus_id in routed:
        roots, _ = _filter_by_tags(corpora[corpus_id], query)
        for node, article_title, source_url in _walk_with_article_metadata(roots):
            if not node.get("text"):
                continue
            score, matched = _score_node(node, query_terms)
            if score > 0:
                candidates.append((score, corpus_id, node, matched, article_title, source_url))

    candidates.sort(key=lambda item: item[0], reverse=True)
    max_score = candidates[0][0] if candidates else 1
    return [
        _snippet_from_node(
            corpus_id=corpus_id,
            node=node,
            score=round(score / max_score, 4),
            reason=f"keyword overlap: {', '.join(matched[:8]) or 'n/a'}",
            article_title=article_title,
            source_url=source_url,
        )
        for score, corpus_id, node, matched, article_title, source_url in candidates[: query.k]
    ], routed


def _load_jsonl(path_name: str) -> tuple[dict[str, Any], ...]:
    path = RAG_DIR / path_name
    if not path.is_file():
        return ()
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return tuple(records)


@lru_cache(maxsize=1)
def load_vector_corpora() -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        corpus_id: records
        for corpus_id, records in {
            "corpus_en": _load_jsonl("vector_corpus_en.jsonl"),
            "corpus_nl": _load_jsonl("vector_corpus_nl.jsonl"),
            "writing_guide": _load_jsonl("vector_writing_guide.jsonl"),
            "schrijfwijzer": _load_jsonl("vector_schrijfwijzer.jsonl"),
        }.items()
        if records
    }


def _normalised_query_vector(query: RetrievalQuery) -> dict[str, float]:
    counts = _token_counts(_query_text(query))
    if not counts:
        return {}
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {term: count / norm for term, count in counts.items()}


def _vector_score(
    record: dict[str, Any],
    query_vector: dict[str, float],
) -> tuple[float, list[str]]:
    terms = record.get("terms") if isinstance(record.get("terms"), dict) else {}
    if not terms or not query_vector:
        return 0.0, []
    matched = sorted(set(terms).intersection(query_vector))
    score = sum(query_vector[term] * float(terms.get(term, 0.0)) for term in matched)
    return score, matched


def _vector_pick(query: RetrievalQuery) -> tuple[list[Snippet], list[str]]:
    vector_corpora = load_vector_corpora()
    article_metadata = load_article_metadata_index()
    routed = _route_corpora(query, set(vector_corpora))
    query_vector = _normalised_query_vector(query)
    candidates: list[tuple[float, str, dict[str, Any], list[str]]] = []
    for corpus_id in routed:
        for record in vector_corpora.get(corpus_id, ()):
            score, matched = _vector_score(record, query_vector)
            if score > 0:
                candidates.append((score, corpus_id, record, matched))

    candidates.sort(key=lambda item: item[0], reverse=True)
    max_score = candidates[0][0] if candidates else 1.0
    snippets: list[Snippet] = []
    for score, corpus_id, record, matched in candidates[: query.k]:
        source_doc = str(record.get("source_doc") or corpus_id)
        node_id = str(record.get("node_id") or record.get("id") or "")
        metadata = article_metadata.get(source_doc, {}).get(node_id, {})
        snippets.append(
            Snippet(
                source_doc=source_doc,
                node_id=node_id,
                title=str(record.get("title") or ""),
                article_title=(
                    _clean_str(record.get("article_title"))
                    or metadata.get("article_title")
                    or _clean_str(record.get("title"))
                ),
                content=str(record.get("content") or ""),
                line_num=record.get("line_num"),
                score=round(score / max_score, 4),
                reason=f"sparse vector overlap: {', '.join(matched[:8]) or 'n/a'}",
                source_url=_clean_str(record.get("source_url")) or metadata.get("source_url"),
            )
        )
    return snippets, routed


class _NodePick(BaseModel):
    node_id: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class _LLMRanking(BaseModel):
    picks: list[_NodePick] = Field(default_factory=list)


_STAGE1_SYSTEM_PROMPT = """You are a retrieval ranker for ABN AMRO's editorial co-pilot.

You see article-level nodes with node_id, title, description, and tags.
Pick the {shortlist_k} articles most likely to contain useful material for the
user's request. Skip articles that are clearly off-topic.

Return each pick with node_id, score from 0.0 to 1.0, and a short reason.
"""

_STAGE2_SYSTEM_PROMPT = """You are a retrieval ranker for ABN AMRO's editorial co-pilot.

You see section trees from articles already shortlisted as relevant. Pick up to
{k} specific sections, or a broad parent only when the parent is better than a
leaf. These snippets will ground the final editorial output.

Return each pick with node_id, score from 0.0 to 1.0, and a short reason.
"""


def _query_context_block(query: RetrievalQuery) -> str:
    return (
        f"User request: {query.user_prompt}\n\n"
        f"Sector: {query.sector or 'N/A'}\n"
        f"Topic keywords: {', '.join(query.topic_keywords) or 'N/A'}\n"
        f"Task codes: {', '.join(query.task_codes) or 'N/A'}\n"
        f"Language: {query.language or 'unspecified'}\n"
        f"Workflow profile ids: {', '.join(query.workflow_profile_ids) or 'N/A'}\n"
        f"Expert profile ids: {', '.join(query.expert_profile_ids) or 'N/A'}\n"
    )


def _render_articles_for_stage1(roots: tuple[dict[str, Any], ...]) -> str:
    lines: list[str] = []
    for node in roots:
        node_id = str(node.get("node_id") or "")
        title = str(node.get("title") or "").strip()
        desc = str(node.get("prefix_summary") or node.get("summary") or "")
        desc = desc.strip().replace("\n", " ")[:280]
        tags = ", ".join(str(tag) for tag in (node.get("tags") or [])) or "-"
        lines.append(f"- [{node_id}] {title}\n    tags: {tags}\n    desc: {desc}")
    return "\n".join(lines)


def _compact_tree(nodes: tuple[dict[str, Any], ...] | list[dict[str, Any]], level: int = 0) -> str:
    lines: list[str] = []
    for node in nodes:
        prefix = "  " * level
        node_id = str(node.get("node_id") or "")
        title = str(node.get("title") or "").strip()
        summary = str(node.get("prefix_summary") or node.get("summary") or "")
        summary = summary.strip().replace("\n", " ")[:180]
        suffix = f" - {summary}" if summary else ""
        lines.append(f"{prefix}- [{node_id}] {title}{suffix}")
        children = node.get("nodes") or []
        if children:
            lines.append(_compact_tree(children, level + 1))
    return "\n".join(line for line in lines if line)


def _stage1_shortlist(
    client: Any,
    model: str,
    *,
    corpus_id: str,
    candidates: tuple[dict[str, Any], ...],
    query: RetrievalQuery,
    shortlist_k: int,
) -> list[_NodePick]:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _STAGE1_SYSTEM_PROMPT.format(shortlist_k=shortlist_k)},
            {
                "role": "user",
                "content": (
                    f"{_query_context_block(query)}\n"
                    f"Corpus: {corpus_id}\n"
                    f"Articles:\n{_render_articles_for_stage1(candidates)}"
                ),
            },
        ],
        response_format=_LLMRanking,
    )
    parsed = completion.choices[0].message.parsed
    return list(parsed.picks) if parsed else []


def _stage2_rank_sections(
    client: Any,
    model: str,
    *,
    corpus_id: str,
    shortlisted: list[dict[str, Any]],
    query: RetrievalQuery,
) -> list[_NodePick]:
    if not shortlisted:
        return []
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _STAGE2_SYSTEM_PROMPT.format(k=query.k)},
            {
                "role": "user",
                "content": (
                    f"{_query_context_block(query)}\n"
                    f"Corpus: {corpus_id}\n"
                    "Shortlisted articles and section tree:\n"
                    f"{_compact_tree(shortlisted)}"
                ),
            },
        ],
        response_format=_LLMRanking,
    )
    parsed = completion.choices[0].message.parsed
    return list(parsed.picks) if parsed else []


def _llm_pick(query: RetrievalQuery, *, api_key: str, model: str) -> tuple[list[Snippet], list[str]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    corpora = load_corpora()
    routed = _route_corpora(query, set(corpora))
    out: list[Snippet] = []
    shortlist_target = max(query.k, min(5, query.k * 2))

    for corpus_id in routed:
        roots = corpora[corpus_id]
        filtered_roots, _ = _filter_by_tags(roots, query)
        nodes_by_id: dict[str, tuple[dict[str, Any], str | None, str | None]] = {}
        for node, article_title, source_url in _walk_with_article_metadata(roots):
            node_id = _clean_str(node.get("node_id"))
            if node_id:
                nodes_by_id[node_id] = (node, article_title, source_url)

        if len(filtered_roots) <= shortlist_target:
            shortlisted = list(filtered_roots)
        else:
            stage1_picks = _stage1_shortlist(
                client,
                model,
                corpus_id=corpus_id,
                candidates=filtered_roots,
                query=query,
                shortlist_k=shortlist_target,
            )
            shortlisted = []
            seen: set[str] = set()
            root_ids = {str(node.get("node_id")) for node in filtered_roots}
            for pick in stage1_picks:
                found = nodes_by_id.get(pick.node_id)
                node = found[0] if found else None
                if node and pick.node_id in root_ids and pick.node_id not in seen:
                    shortlisted.append(node)
                    seen.add(pick.node_id)
            if not shortlisted:
                shortlisted = list(filtered_roots)

        section_picks = _stage2_rank_sections(
            client,
            model,
            corpus_id=corpus_id,
            shortlisted=shortlisted,
            query=query,
        )
        for pick in section_picks:
            found = nodes_by_id.get(pick.node_id)
            if not found:
                continue
            node, article_title, source_url = found
            if not node or not node.get("text"):
                continue
            out.append(
                _snippet_from_node(
                    corpus_id=corpus_id,
                    node=node,
                    score=round(max(0.0, min(1.0, pick.score)), 4),
                    reason=pick.reason,
                    article_title=article_title,
                    source_url=source_url,
                )
            )

    out.sort(key=lambda snippet: snippet.score, reverse=True)
    return out[: query.k], routed


def retrieve_context(
    query: RetrievalQuery,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> RetrievalResult:
    if query.retrieval_backend == "vector_rag":
        snippets, routed = _vector_pick(query)
        if snippets:
            return RetrievalResult(
                query=query,
                snippets=snippets,
                provider=query.retrieval_backend,
                corpora_searched=routed,
                source="deterministic",
                reasoning="Sparse vector cosine over generated JSONL chunk assets.",
            )
        fallback_snippets, fallback_routed = _deterministic_pick(query)
        return RetrievalResult(
            query=query,
            snippets=fallback_snippets,
            provider=query.retrieval_backend,
            corpora_searched=routed or fallback_routed,
            source="deterministic",
            reasoning="Vector assets returned no snippets; PageIndex keyword fallback used.",
        )

    fallback_reason = ""
    if api_key and model:
        try:
            snippets, routed = _llm_pick(query, api_key=api_key, model=model)
            if snippets:
                return RetrievalResult(
                    query=query,
                    snippets=snippets,
                    provider=query.retrieval_backend,
                    corpora_searched=routed,
                    model=model,
                    source="llm",
                    reasoning="Two-stage LLM ranker shortlisted articles and selected grounding sections.",
                )
            fallback_reason = "LLM ranker returned no usable snippets; deterministic fallback."
        except Exception as exc:
            fallback_reason = f"LLM ranker failed; deterministic fallback: {exc}"

    snippets, routed = _deterministic_pick(query)
    return RetrievalResult(
        query=query,
        snippets=snippets,
        provider=query.retrieval_backend,
        corpora_searched=routed,
        source="deterministic",
        reasoning=fallback_reason or "Keyword and tag overlap over cached PageIndex assets.",
    )
