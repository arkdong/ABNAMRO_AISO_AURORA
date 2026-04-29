# Experiments — chunking × embedding

Goal: find a chunking strategy + embedding model combo that maximises
retrieval precision on AURORA's mixed Dutch/English insight content.

## How we measure

- **Eval set:** 10–15 fixed queries with hand-labeled relevant chunks.
  Seed list lives in `tests/test_retrieval.py`; expand as we ingest more data.
- **Primary metric:** precision@5 — of the 5 chunks returned, how many are
  actually relevant.
- **Secondary:** recall@10, MRR (mean reciprocal rank), latency per query.
- **Protocol:** for every (chunker, model) pair, re-index from scratch into a
  fresh Chroma collection (never mix models in one collection), then run all
  eval queries and record metrics in the table below.

For more advanced chunking strategies (contextual retrieval, late chunking,
RAPTOR, small-to-big, HyDE-indexing, etc.) see
[`advanced-chunking.md`](advanced-chunking.md).
For embedding-side techniques that wrap a base model (rerankers, hybrid
BM25 + dense, HyDE-at-query, multi-query / RRF, ColBERT, etc.) see
[`advanced-embedding.md`](advanced-embedding.md).

## Chunking strategies to try

| ID  | Name                | Description                                                                 | Applies to            |
|-----|---------------------|-----------------------------------------------------------------------------|-----------------------|
| C1  | Fixed-size 512/50   | Recursive splitter, 512 tokens, 50 overlap                                  | articles              |
| C2  | Fixed-size 256/25   | Smaller window — tests whether tighter chunks help precision                | articles              |
| C3  | Fixed-size 1024/100 | Larger window — tests whether more context helps                            | articles              |
| C4  | Paragraph           | Split on `\n\n`, drop fragments < 50 chars                                  | articles              |
| C5  | Sentence-window     | One sentence per chunk + N neighbouring sentences as context                | articles              |
| C6  | Semantic            | Split where embedding similarity between adjacent sentences drops           | articles              |
| C7  | Section heading     | Split at numbered `1.2.3` headings, one rule per chunk                      | writing guide         |
| C8  | Heading + parent    | Each rule chunk also carries its parent section title in metadata           | writing guide         |
| C9  | Sliding window      | Overlapping windows over headings (catches rules that span sections)        | writing guide         |
| C10 | Article-as-chunk    | Whole article as one chunk (baseline, expected weak)                        | articles              |

## Embedding models to try

| ID  | Model                                          | Lang     | Dim   | Notes                                |
|-----|------------------------------------------------|----------|-------|--------------------------------------|
| E1  | `intfloat/multilingual-e5-large`               | multi    | 1024  | Recommended baseline, NL+EN          |
| E2  | `intfloat/multilingual-e5-base`                | multi    | 768   | Smaller / faster comparison          |
| E3  | `sentence-transformers/all-MiniLM-L6-v2`       | EN       | 384   | Fast English baseline                |
| E4  | `BAAI/bge-m3`                                  | multi    | 1024  | Strong multilingual, dense + sparse  |
| E5  | `BAAI/bge-large-en-v1.5`                       | EN       | 1024  | English-only ceiling test            |
| E6  | `jinaai/jina-embeddings-v3`                    | multi    | 1024  | Recent multilingual model            |
| E7  | OpenAI `text-embedding-3-large`                | multi    | 3072  | API-based ceiling, costs money       |
| E8  | OpenAI `text-embedding-3-small`                | multi    | 1536  | Cheaper OpenAI option                |
| E9  | Cohere `embed-multilingual-v3.0`               | multi    | 1024  | API-based, NL-aware                  |

## Translation axis

For each (chunker, model) pair we also vary the input language:
- **T1:** English-translated body (default for English-only models)
- **T2:** Original Dutch body (only for multilingual models)

This answers research question #1.

## Results table

Fill one row per run. Keep it append-only; don't delete old rows.

| Date       | Chunker | Embedder | Lang | Eval queries | P@5 | R@10 | MRR | Notes |
|------------|---------|----------|------|--------------|-----|------|-----|-------|
| _pending_  | —       | —        | —    | —            | —   | —    | —   | —     |

## Decisions log

Record any "we picked X over Y because Z" calls here so we don't relitigate.

- _(empty)_
