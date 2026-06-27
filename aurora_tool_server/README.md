# AURORA Tool Server

Standalone AURORA backend exposing ABN AMRO editorial grounding and evaluation
as Python core services, a REST API, MCP tools, and a Streamlit frontend.

This package is the final runnable product. It does not import from the removed
legacy `archive/` tree. Runtime assets are bundled under `assets/`, while the
repository root `data/` folder is retained only as rebuild/source material.

## Run

Start the REST API from this directory:

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

The AI Agent page uses the OpenAI Agents SDK. You can also set the key in the
shell and optionally override `AURORA_AGENT_MODEL`:

```bash
OPENAI_API_KEY=sk-... uv run streamlit run frontend/app.py
```

## REST Tool Surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/runs` | Run the full AURORA pipeline |
| `POST` | `/v1/intent/classify` | Classify task, role, sector, keywords, and language |
| `POST` | `/v1/profiles/select` | Select workflow and domain expert profiles |
| `POST` | `/v1/retrieval/search` | Retrieve approved grounding snippets |
| `POST` | `/v1/prompts/refine` | Convert a weak request into a grounded instruction |
| `POST` | `/v1/drafts/generate` | Generate a source-backed draft |
| `POST` | `/v1/evaluations/score` | Score a draft against the KPI catalogue |
| `GET` | `/v1/runs/{run_id}/audit` | Return the run audit trace |

Profile CRUD endpoints remain available at `/v1/profiles` for admin workflows.

## MCP Tool Surface

- `aurora_classify_intent`
- `aurora_select_profiles`
- `aurora_retrieve_context`
- `aurora_refine_prompt`
- `aurora_generate_draft`
- `aurora_evaluate_draft`
- `aurora_run_pipeline`
- `aurora_get_audit_trace`

Run the MCP module directly when an MCP host needs a local server process:

```bash
uv run python -m aurora_tool_server.mcp_server
```

## Runtime Assets

The live server reads from:

- `assets/rag/` for PageIndex and local sparse-vector retrieval assets.
- `assets/profiles/` for workflow and domain expert profiles.
- `assets/evaluation/kpi_catalogue.json` for KPI scoring.
- `assets/prompts/` for generation and evaluation system prompts.

The root `data/` folder and root guide PDFs are retained for rebuilds. They are
not required by the live runtime path.

Audit events are kept in memory for `/v1/runs/{run_id}/audit` and also emitted
to stdout plus `logs/audit.jsonl` by default. Override the file path with
`AURORA_AUDIT_LOG_PATH`.

## Test

```bash
uv run pytest
uv run python -m py_compile aurora_tool_server/core.py aurora_tool_server/api.py aurora_tool_server/mcp_server.py
uv run python -m py_compile frontend/app.py frontend/api_client.py frontend/agent_tools.py frontend/agent_service.py frontend/pages/1_Pipeline_Inspector.py frontend/pages/2_Normal_Mode.py frontend/pages/3_Settings.py frontend/pages/4_Profile.py
```
