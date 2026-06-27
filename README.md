# AURORA ABN AMRO Submission

AURORA is a governed editorial grounding and evaluation layer for ABN AMRO
content workflows. The final runnable product lives in `aurora_tool_server/`
and exposes the core pipeline as Python services, REST endpoints, MCP tools,
and a small Streamlit frontend.

## Repository Shape

- `aurora_tool_server/`: final application code, tests, runtime assets, REST
  API, MCP server, and Streamlit frontend.
- `data/`: retained source material for transparency and future RAG/KPI rebuilds.
- `deliverables/`: final submission PDFs.
- `Dockerfile.railway` and `railway.toml`: root-level deployment entrypoints
  that build only the live tool server package.

The old PoC archive, generated output, temporary PDF renders, root `rag/`
artifacts, context notes, caches, and workshop-only interfaces were removed.

## Run Locally

From `aurora_tool_server/`:

```bash
uv run uvicorn aurora_tool_server.api:app --reload
```

Open REST docs at `http://127.0.0.1:8000/docs`.

In a second terminal:

```bash
uv run streamlit run frontend/app.py
```

Set `OPENAI_API_KEY` in `aurora_tool_server/.env` for LLM-backed stages. Without
it, the server uses deterministic local fallbacks where available.

## Core REST Tools

- `POST /v1/runs`
- `POST /v1/intent/classify`
- `POST /v1/profiles/select`
- `POST /v1/retrieval/search`
- `POST /v1/prompts/refine`
- `POST /v1/drafts/generate`
- `POST /v1/evaluations/score`
- `GET /v1/runs/{run_id}/audit`

Profile CRUD endpoints remain available for admin-style profile management.

## Core MCP Tools

- `aurora_classify_intent`
- `aurora_select_profiles`
- `aurora_retrieve_context`
- `aurora_refine_prompt`
- `aurora_generate_draft`
- `aurora_evaluate_draft`
- `aurora_run_pipeline`
- `aurora_get_audit_trace`

## Runtime Assets And Source Data

Runtime retrieval and evaluation use assets under:

- `aurora_tool_server/assets/rag/`
- `aurora_tool_server/assets/profiles/`
- `aurora_tool_server/assets/evaluation/`
- `aurora_tool_server/assets/prompts/`

The root `data/` folder and guide PDFs are kept so the RAG assets can be rebuilt
later. The live server does not import from old archive code.

## Verification

From `aurora_tool_server/`:

```bash
uv run pytest
uv run python -m py_compile aurora_tool_server/core.py aurora_tool_server/api.py aurora_tool_server/mcp_server.py
uv run python -m py_compile frontend/app.py frontend/api_client.py frontend/agent_tools.py frontend/agent_service.py frontend/pages/1_Pipeline_Inspector.py frontend/pages/2_Normal_Mode.py frontend/pages/3_Settings.py frontend/pages/4_Profile.py
```
