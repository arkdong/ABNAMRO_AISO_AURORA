"""PageIndex-backed RAG provider.

Bypasses ``rag.pageindex.PageIndexClient`` and its workspace machinery; instead
loads the cached structure JSONs directly via :func:`corpus_loader.load_corpora`.
Two ranking paths:

- LLM: a **two-stage** ranker. Stage 1 sees only top-level (article) nodes
  with title/prefix_summary/tags and shortlists a handful of articles. Stage 2
  ranks sections within just those articles. Falls back to deterministic on
  any failure.
- Deterministic: token-overlap scoring against title + summary.

A **tag-aware pre-filter** runs ahead of both paths: when ``topic_keywords``
or ``sector`` overlap with the article-level ``tags`` baked into the tree
(see ``rag/scripts/enrich_structure.py``), the candidate set is narrowed to
the tagged articles — but only when that still leaves enough candidates to
fill ``query.k``. Tags are Dutch (corpus origin) and queries may be English,
so the deterministic title/summary scoring remains the safety net.

The vendored PageIndex's ``get_page_content`` is not used — every node already
carries its ``text`` inline, so we just look the picked node up by id.
"""

from __future__ import annotations

import os
import re

import openai
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

from backend.retrieval.corpus_loader import CorpusDoc, load_corpora, walk_nodes
from backend.retrieval.types import RetrievalQuery, RetrievalResult, Snippet

# Task code → which corpora to consult. Multi-intent unions the values.
_CORPUS_ROUTING: dict[str, tuple[str, ...]] = {
    "T1_DRAFT": ("corpus_en", "writing_guide"),
    "T1_SEARCH": ("corpus_en",),
    "T1_TRANSLATE": ("corpus_en",),
    "T2_COMPLIANCE": ("writing_guide", "corpus_en"),
    "T4_RENEWAL": ("corpus_en", "writing_guide"),
}


def _route_corpora(task_codes: list[str], available: set[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for code in task_codes or ["T1_DRAFT"]:
        for doc_id in _CORPUS_ROUTING.get(code, ("corpus_en",)):
            if doc_id in available and doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out


# ── Tag-aware pre-filter ────────────────────────────────────────────────────


def _query_terms(query: RetrievalQuery) -> list[str]:
    """Surface the soft signals we want to match against article tags.

    Only ``topic_keywords`` participate — ``sector`` is too coarse for tag
    narrowing in a single-sector corpus (it matches every article), but it
    still goes into the LLM prompt as context.
    """
    return [kw.lower() for kw in query.topic_keywords if kw]


def _tag_matches(node: dict, terms: list[str]) -> bool:
    tags = [str(t).lower() for t in (node.get("tags") or [])]
    if not tags or not terms:
        return False
    for term in terms:
        for tag in tags:
            if term in tag or tag in term:
                return True
    return False


def _filter_by_tags(
    nodes: tuple[dict, ...],
    query: RetrievalQuery,
) -> tuple[tuple[dict, ...], bool]:
    """Narrow top-level nodes to those whose tags overlap query terms.

    Narrows whenever at least one article matched a tag and the matched set is
    strictly smaller than the input. We don't require ``len(matched) >= k`` —
    each article has multiple sections, so stage 2 can still produce ``k``
    snippets from just a couple of articles. Returns ``(filtered, narrowed)``.
    """
    terms = _query_terms(query)
    if not terms:
        return nodes, False
    matched = tuple(n for n in nodes if _tag_matches(n, terms))
    if 0 < len(matched) < len(nodes):
        return matched, True
    return nodes, False


# ── Deterministic scoring ───────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+")

# Without this, English stopwords (the/and/for/are…) in the writing_guide's
# tiny section titles outscore real topical overlap in the corpus_en articles.
_STOPWORDS: frozenset[str] = frozenset({
    # EN
    "the", "and", "for", "but", "with", "this", "that", "from", "are", "was",
    "were", "you", "your", "our", "can", "will", "what", "when", "where",
    "why", "how", "who", "has", "have", "had", "not", "all", "any", "also",
    "into", "out", "off", "its", "their", "they", "them", "than", "then",
    "such", "some", "more", "most", "much", "many", "very", "just", "only",
    "use", "used", "using", "make", "made", "get", "got", "see", "show",
    "shown", "let", "lets", "via", "about", "above", "below", "after",
    "before", "between", "over", "under", "while",
    # NL
    "het", "een", "niet", "voor", "met", "van", "zijn", "waren", "aan",
    "naar", "dat", "deze", "die", "ook", "maar", "door", "want", "omdat",
    "dan", "nog", "wel", "wat", "wie", "waar", "hoe", "als", "bij", "tot",
    "uit", "alleen", "meer", "meest",
})


def _tokens(text: str, min_len: int = 3) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= min_len and t.lower() not in _STOPWORDS
    }


def _node_score(node: dict, query_tokens: set[str]) -> int:
    title_tokens = _tokens(node.get("title", ""))
    summary_blob = (node.get("prefix_summary") or "") + " " + (node.get("summary") or "")
    summary_tokens = _tokens(summary_blob)
    return 3 * len(title_tokens & query_tokens) + len(summary_tokens & query_tokens)


def _query_token_set(query: RetrievalQuery) -> set[str]:
    """Build the bag of tokens to match against node title/summary. We tokenize
    ``topic_keywords`` at ``min_len=2`` so short acronyms like ``AI`` survive
    (otherwise multi-word keywords like ``agentic AI`` would collapse to a
    single never-matching token)."""
    return _tokens(query.user_prompt) | _tokens(
        " ".join(query.topic_keywords), min_len=2
    )


def _deterministic_pick(
    corpora: dict[str, CorpusDoc],
    corpus_ids: list[str],
    query: RetrievalQuery,
) -> list[Snippet]:
    qtokens = _query_token_set(query)
    candidates: list[tuple[int, str, dict, list[str]]] = []
    for doc_id in corpus_ids:
        roots, _ = _filter_by_tags(corpora[doc_id].nodes, query)
        for node in walk_nodes(roots):
            if not node.get("text"):
                continue
            score = _node_score(node, qtokens)
            if score > 0:
                matched = sorted(
                    _tokens(node.get("title", "") + " " +
                            (node.get("prefix_summary") or "") + " " +
                            (node.get("summary") or "")) & qtokens
                )
                candidates.append((score, doc_id, node, matched))
    candidates.sort(key=lambda t: -t[0])
    if not candidates:
        return []
    max_score = candidates[0][0]
    out: list[Snippet] = []
    for score, doc_id, node, matched in candidates[: query.k]:
        out.append(
            Snippet(
                source_doc=doc_id,
                node_id=node.get("node_id", ""),
                title=node.get("title", ""),
                content=node.get("text", ""),
                line_num=node.get("line_num") or node.get("page_index"),
                score=score / max_score,
                reason=f"keyword overlap: {', '.join(matched[:6]) or 'n/a'}",
            )
        )
    return out


# ── LLM ranking (two-stage) ─────────────────────────────────────────────────


class _NodePick(BaseModel):
    node_id: str
    score: float
    reason: str


class _LLMRanking(BaseModel):
    picks: list[_NodePick] = Field(default_factory=list)


_STAGE1_SYSTEM_PROMPT = """You are a retrieval ranker for ABN AMRO's editorial co-pilot.
You see a list of articles (top-level nodes). For each article you get its
node_id, title, a brief description, and the topic tags it was published under.

Pick the {shortlist_k} articles most likely to contain useful material for the
user's request. Skip articles that are clearly off-topic.

Return each pick with:
- node_id: must exactly match one in the list
- score: article-level relevance 0.0–1.0
- reason: one short sentence
"""

_STAGE2_SYSTEM_PROMPT = """You are a retrieval ranker for ABN AMRO's editorial co-pilot.
You see the section trees of a small set of articles already shortlisted as
relevant. Pick up to {k} specific sections (or whole articles) most useful for
the user's request.

Return each pick with:
- node_id: must exactly match one in the trees
- score: section relevance 0.0–1.0
- reason: one short sentence

Prefer specific leaves over broad parent nodes when the leaf already covers the
ask; otherwise pick the parent.
"""


def _compact_tree(nodes, level: int = 0) -> str:
    """Render the tree as id/title (+summary) lines — no full text body."""
    lines: list[str] = []
    for n in nodes:
        prefix = "  " * level
        title = (n.get("title") or "").strip()
        nid = n.get("node_id", "")
        summary = ((n.get("prefix_summary") or n.get("summary") or "")
                   .strip().replace("\n", " "))[:160]
        suffix = f" — {summary}" if summary else ""
        lines.append(f"{prefix}- [{nid}] {title}{suffix}")
        if n.get("nodes"):
            child = _compact_tree(n["nodes"], level + 1)
            if child:
                lines.append(child)
    return "\n".join(lines)


def _render_articles_for_stage1(roots: tuple[dict, ...]) -> str:
    """One line per article: id, title, description (truncated), tags."""
    lines: list[str] = []
    for n in roots:
        nid = n.get("node_id", "")
        title = (n.get("title") or "").strip()
        desc = ((n.get("prefix_summary") or n.get("summary") or "")
                .strip().replace("\n", " "))[:280]
        tags = ", ".join(str(t) for t in (n.get("tags") or [])) or "—"
        lines.append(f"- [{nid}] {title}\n    tags: {tags}\n    desc: {desc}")
    return "\n".join(lines)


def _query_context_block(query: RetrievalQuery) -> str:
    return (
        f"User request: {query.user_prompt}\n\n"
        f"Sector: {query.sector or 'N/A'}\n"
        f"Topic keywords: {', '.join(query.topic_keywords) or 'N/A'}\n"
        f"Task codes: {', '.join(query.task_codes) or 'N/A'}\n"
        f"Language: {query.language or 'unspecified'}\n"
    )


def _stage1_shortlist(
    client: "openai.OpenAI",
    model: str,
    doc: CorpusDoc,
    candidates: tuple[dict, ...],
    query: RetrievalQuery,
    shortlist_k: int,
) -> list[_NodePick]:
    """Ask the LLM to pick ``shortlist_k`` article-level nodes."""
    user_msg = (
        f"{_query_context_block(query)}\n"
        f"Document: {doc.doc_name}\n"
        f"Articles:\n{_render_articles_for_stage1(candidates)}"
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _STAGE1_SYSTEM_PROMPT.format(shortlist_k=shortlist_k)},
            {"role": "user", "content": user_msg},
        ],
        response_format=_LLMRanking,
    )
    parsed = completion.choices[0].message.parsed
    return list(parsed.picks) if parsed else []


def _stage2_rank_sections(
    client: "openai.OpenAI",
    model: str,
    doc: CorpusDoc,
    shortlisted: list[dict],
    query: RetrievalQuery,
    k: int,
) -> list[_NodePick]:
    """Send the full subtree of each shortlisted article and rank sections."""
    if not shortlisted:
        return []
    tree_str = _compact_tree(tuple(shortlisted))
    user_msg = (
        f"{_query_context_block(query)}\n"
        f"Document: {doc.doc_name}\n"
        f"Shortlisted articles and their sections:\n{tree_str}"
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _STAGE2_SYSTEM_PROMPT.format(k=k)},
            {"role": "user", "content": user_msg},
        ],
        response_format=_LLMRanking,
    )
    parsed = completion.choices[0].message.parsed
    return list(parsed.picks) if parsed else []


def _llm_pick(
    corpora: dict[str, CorpusDoc],
    corpus_ids: list[str],
    query: RetrievalQuery,
    api_key: str,
    model: str,
) -> list[Snippet]:
    """Two-stage LLM ranker. See module docstring for the contract."""
    client = openai.OpenAI(api_key=api_key)
    out: list[Snippet] = []
    # Stage 1's shortlist size: aim for ~2x final k so stage 2 has options,
    # capped by how many articles actually exist after tag filtering.
    shortlist_target = max(query.k, min(5, query.k * 2))

    for doc_id in corpus_ids:
        doc = corpora[doc_id]
        all_roots = doc.nodes
        filtered_roots, narrowed = _filter_by_tags(all_roots, query)
        if narrowed:
            logger.info(
                f"PageIndex tag pre-filter: {doc_id} narrowed "
                f"{len(all_roots)} → {len(filtered_roots)} articles"
            )

        nodes_by_id = {
            n.get("node_id"): n for n in walk_nodes(all_roots) if n.get("node_id")
        }

        # Skip stage 1 when there's nothing to shrink (few articles, ranker can
        # see the full tree just fine).
        if len(filtered_roots) <= shortlist_target:
            shortlisted = list(filtered_roots)
        else:
            stage1_picks = _stage1_shortlist(
                client, model, doc, filtered_roots, query, shortlist_target
            )
            shortlisted = []
            seen: set[str] = set()
            for pick in stage1_picks:
                node = nodes_by_id.get(pick.node_id)
                if node and pick.node_id not in seen and node in filtered_roots:
                    shortlisted.append(node)
                    seen.add(pick.node_id)
            # If the LLM returns garbage, don't crash — fall back to the full
            # filtered set for stage 2 (still smaller than the original).
            if not shortlisted:
                logger.warning(
                    f"PageIndex stage 1 returned no usable picks for {doc_id}; "
                    "passing full filtered set to stage 2"
                )
                shortlisted = list(filtered_roots)

        section_picks = _stage2_rank_sections(
            client, model, doc, shortlisted, query, query.k
        )

        for pick in section_picks:
            node = nodes_by_id.get(pick.node_id)
            if not node or not node.get("text"):
                continue
            out.append(
                Snippet(
                    source_doc=doc_id,
                    node_id=pick.node_id,
                    title=node.get("title", ""),
                    content=node.get("text", ""),
                    line_num=node.get("line_num") or node.get("page_index"),
                    score=max(0.0, min(1.0, pick.score)),
                    reason=pick.reason,
                )
            )

    out.sort(key=lambda s: -s.score)
    return out[: query.k]


# ── Provider ────────────────────────────────────────────────────────────────


class PageIndexProvider:
    name = "pageindex"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        corpora: dict[str, CorpusDoc] | None = None,
    ):
        # When no key is passed explicitly, fall back to the PageIndex-scoped
        # env var. This keeps the retrieval LLM call decoupled from any
        # shared/intent key the caller might be using elsewhere.
        self._api_key = api_key or os.getenv("OPENAI_API_KEY_PAGEINDEX")
        self._model = model
        self._corpora = corpora if corpora is not None else load_corpora()

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        corpus_ids = _route_corpora(query.task_codes, set(self._corpora.keys()))
        if not corpus_ids:
            return RetrievalResult(
                snippets=[],
                provider=self.name,
                corpora_searched=[],
                source="deterministic",
            )
        if self._api_key and self._model:
            try:
                logger.info(f"PageIndex retrieval via LLM (model={self._model}, corpora={corpus_ids})")
                snippets = _llm_pick(
                    self._corpora, corpus_ids, query, self._api_key, self._model
                )
                return RetrievalResult(
                    snippets=snippets,
                    provider=self.name,
                    corpora_searched=corpus_ids,
                    source="llm",
                )
            except Exception as e:
                logger.warning(
                    f"PageIndex LLM ranker failed ({e}); falling back to deterministic"
                )
        snippets = _deterministic_pick(self._corpora, corpus_ids, query)
        return RetrievalResult(
            snippets=snippets,
            provider=self.name,
            corpora_searched=corpus_ids,
            source="deterministic",
        )
