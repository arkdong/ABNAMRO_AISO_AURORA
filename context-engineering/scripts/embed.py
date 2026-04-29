"""Embed chunks and store them in a local Chroma collection.

Use the SAME model for indexing and querying — never mix.
"""

from __future__ import annotations

from pathlib import Path

VECTOR_DB_DIR = Path(__file__).resolve().parent.parent / "vector_db"
COLLECTION_NAME = "aurora_mvp"

DEFAULT_MODEL = "intfloat/multilingual-e5-large"


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True).tolist()


def add_chunks(texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    collection = get_collection()
    embeddings = embed_texts(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def main() -> None:
    raise NotImplementedError("Wire up after chunk.py produces data/chunked/*.json.")


if __name__ == "__main__":
    main()
