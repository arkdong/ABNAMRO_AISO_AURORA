"""Retrieval data shapes (Pydantic)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """Everything a RAG provider might want from the upstream stages."""

    user_prompt: str
    task_codes: list[str] = Field(default_factory=list)
    sector: Optional[str] = None
    topic_keywords: list[str] = Field(default_factory=list)
    language: Optional[Literal["en", "nl", "both"]] = None
    workflow_profile_ids: list[str] = Field(default_factory=list)
    expert_profile_ids: list[str] = Field(default_factory=list)
    k: int = 5


class Snippet(BaseModel):
    source_doc: str
    node_id: str
    title: str
    content: str
    line_num: Optional[int] = None
    score: float
    reason: str


class RetrievalResult(BaseModel):
    snippets: list[Snippet] = Field(default_factory=list)
    provider: str
    corpora_searched: list[str] = Field(default_factory=list)
    source: Literal["llm", "deterministic"]
