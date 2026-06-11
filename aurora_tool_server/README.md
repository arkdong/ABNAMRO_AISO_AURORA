# AURORA Tool Server

Standalone phase-1 AURORA backend exposing the editorial grounding pipeline as
Python core services, a REST API, and MCP tools.

This package intentionally does not import from `../archive`. Runtime assets are
copied into `assets/` so the server can run independently.

## Run

```bash
uv run uvicorn aurora_tool_server.api:app --reload
```

REST docs are available at `/docs`, and the OpenAPI schema is available at
`/openapi.json`.

In a second terminal, start the server-backed Streamlit frontend:

```bash
uv run streamlit run frontend/app.py
```

The API server and Streamlit app load `./.env` automatically. Put your key in:

```bash
OPENAI_API_KEY=sk-...
```

With `OPENAI_API_KEY` set, the pipeline uses LLM paths for intent
classification, profile selection, PageIndex retrieval reranking, prompt
refinement, draft generation, and evaluation. Each stage falls back to the
local deterministic path if its LLM call fails or is not configured.

Optional per-stage overrides are available in `.env.example`, including:
`OPENAI_API_KEY_PROFILE_SELECTION`, `OPENAI_API_KEY_PAGEINDEX`,
`OPENAI_API_KEY_PROMPT_REFINEMENT`, `OPENAI_API_KEY_CONTENT_GENERATION`, and
`OPENAI_API_KEY_EVALUATION`, plus matching `AURORA_*_MODEL` values.

The AI Agent Interface page uses the OpenAI Agents SDK. You can also set the
key in the shell and optionally override `AURORA_AGENT_MODEL`:

```bash
OPENAI_API_KEY=sk-... uv run streamlit run frontend/app.py
```

## Test

```bash
uv run pytest
uv run python -m py_compile aurora_tool_server/core.py aurora_tool_server/api.py aurora_tool_server/mcp_server.py
uv run python -m py_compile frontend/app.py frontend/api_client.py frontend/agent_tools.py frontend/agent_service.py frontend/pages/2_AI_Agent_Interface.py
```
