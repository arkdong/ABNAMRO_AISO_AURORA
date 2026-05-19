"""Retrieval orchestration.

- :func:`build_query` turns ``(user_prompt, IntentResult, ProfileBundle)`` into
  a :class:`RetrievalQuery`.
- :func:`retrieve` fans the query out to one or more providers, merging by
  score and capping at ``query.k``.
"""

from __future__ import annotations

from backend.intent import IntentResult
from backend.retrieval.provider import RagProvider
from backend.retrieval.types import RetrievalQuery, RetrievalResult
from profiles import ProfileBundle


def build_query(
    user_prompt: str,
    intent: IntentResult,
    bundle: ProfileBundle,
    k: int = 5,
) -> RetrievalQuery:
    return RetrievalQuery(
        user_prompt=user_prompt,
        task_codes=list(intent.task_codes),
        sector=intent.sector,
        topic_keywords=list(intent.topic_keywords),
        language=intent.language,
        workflow_profile_ids=[w.id for w in bundle.workflow],
        expert_profile_ids=[e.id for e in bundle.domain_expert],
        k=k,
    )


def retrieve(
    query: RetrievalQuery,
    providers: list[RagProvider] | None = None,
) -> RetrievalResult:
    if not providers:
        from backend.retrieval.pageindex_provider import PageIndexProvider

        providers = [PageIndexProvider()]
    if len(providers) == 1:
        return providers[0].retrieve(query)

    all_snippets = []
    corpora_searched: list[str] = []
    sources: set[str] = set()
    provider_names: list[str] = []
    for p in providers:
        r = p.retrieve(query)
        all_snippets.extend(r.snippets)
        for c in r.corpora_searched:
            if c not in corpora_searched:
                corpora_searched.append(c)
        sources.add(r.source)
        provider_names.append(r.provider)

    all_snippets.sort(key=lambda s: -s.score)
    return RetrievalResult(
        snippets=all_snippets[: query.k],
        provider="+".join(provider_names),
        corpora_searched=corpora_searched,
        source="llm" if "llm" in sources else "deterministic",
    )
