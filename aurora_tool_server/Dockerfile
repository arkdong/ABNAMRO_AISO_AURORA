FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY aurora_tool_server ./aurora_tool_server
COPY frontend ./frontend
COPY assets ./assets
COPY deploy ./deploy

RUN uv sync --locked --no-dev

EXPOSE 8501

CMD ["sh", "deploy/start.sh"]

