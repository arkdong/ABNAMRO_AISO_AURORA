"""Retrieval quality tests.

Run a fixed set of queries against the indexed Chroma collection and record
precision@5 — how many of the top 5 retrieved chunks are manually marked relevant.
Used to compare chunking strategies and embedding models.
"""

from __future__ import annotations

TEST_QUERIES = [
    # (query, expected_sector)
    ("rising interest rates impact on housing market", "real_estate"),
    ("nitrogen rules and dairy farming", "agriculture"),
    ("solar panel financing for SMEs", "energy"),
]


def test_placeholder() -> None:
    """Replace with real retrieval-quality assertions once data is indexed."""
    assert TEST_QUERIES
