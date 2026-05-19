# One-time setup: build the indexes the generate pipeline needs.
# Articles (a9 × gpt-5): dense vectors + BM25.
# Writing guide (a10 × writing-guide): BM25 (dense vectors built by the embed step above).
# Requires: pip install rank-bm25
python -m scripts.embed --embedder e4
python -m scripts.embed --embedder e4 --chunker a9  --src gpt-5
python -m scripts.embed --embedder x4 --chunker a9  --src gpt-5
python -m scripts.embed --embedder x4 --chunker a10-raptor-structural --src writing-guide

# Generate
# Dry run
python -m scripts.generate \
    --query "I want to write about how generative AI is changing advertising for Dutch businesses" \
    --top-k-articles 3 --top-k-rules 5 \
    --llm-provider anthropic \
	--dry-run

# With API key
set -a; source .env; set +a && \
python -m scripts.generate \
    --query "I want to write about how generative AI is changing advertising for Dutch businesses" \
    --top-k-articles 3 --top-k-rules 5 \
    --llm-provider openai --model gpt-4o \
	--show-context --show-prompt
