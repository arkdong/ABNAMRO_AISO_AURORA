"""Dual retrieval: style references from articles + writing rules from the guide."""

from __future__ import annotations

from .embed import DEFAULT_MODEL, get_collection


def embed_query(text: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).encode(text).tolist()


def retrieve_style_references(query: str, sector: str | None, top_k: int = 3) -> dict:
    where = {"doc_type": "article"}
    if sector:
        where["sector"] = sector
    return get_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where=where,
    )


def retrieve_writing_rules(query: str, top_k: int = 10) -> dict:
    return get_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where={"doc_type": "writing_guide"},
    )


def retrieve(query: str, sector: str | None) -> dict:
    return {
        "style_references": retrieve_style_references(query, sector),
        "writing_rules": retrieve_writing_rules(query),
    }


if __name__ == "__main__":
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else input("Query: ")
    sector = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(retrieve(q, sector), indent=2, default=str))
