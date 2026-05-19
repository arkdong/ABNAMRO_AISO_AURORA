"""RagProvider protocol — the seam for swapping retrieval backends.

Future implementations (vector, hybrid, agentic PageIndex) only need to honour
``retrieve(query) -> RetrievalResult`` and carry a ``name`` attribute.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.retrieval.types import RetrievalQuery, RetrievalResult


@runtime_checkable
class RagProvider(Protocol):
    name: str

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...
