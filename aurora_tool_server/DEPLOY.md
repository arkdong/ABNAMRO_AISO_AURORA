# Deploy AURORA Agent

This deployment runs the full standalone AURORA experience from `aurora_tool_server/`:

- FastAPI tool server on private container port `8000`
- Streamlit frontend on the public `$PORT`
- AI Agent Interface Streamlit page
- PageIndex context retrieval from `assets/rag/`
- Product-neutral audit events on stdout and, best-effort, `logs/audit.jsonl`

## Recommended Host

Use a Docker-based web service on Render or Railway.

The deployable standalone app is `aurora_tool_server/`. The legacy `archive/`
tree and root generated RAG artifacts are intentionally absent from the final
submission. The root `data/` folder is retained for rebuild transparency, but
the Docker image uses the prebuilt runtime assets inside `aurora_tool_server/`.

## Railway From Repo Root

Railway can deploy directly from the repository root. The repo-root
`railway.toml` points Railway at `Dockerfile.railway`, and the repo-root
`.dockerignore` keeps the build context focused on this standalone package.

1. Create a Railway service from the GitHub repo.
2. Do not set a custom root directory.
3. Leave `PORT` unset so Railway injects it.
4. Add the secrets below.
5. Deploy, then generate a public domain.
6. If Railway asks for a target port, set it to the port shown in the runtime
   log line `Starting Streamlit on 0.0.0.0:...`, or remove and regenerate the
   domain so it follows Railway's injected `PORT`.

## Local Railway Docker Check

From the repository root:

```bash
docker build -f Dockerfile.railway -t aurora-agent .
docker run --env OPENAI_API_KEY="$OPENAI_API_KEY" -p 8501:8501 aurora-agent
```

Open `http://localhost:8501`.

## Required Secrets

Set these in the host's environment/secrets UI. Do not commit `.env`.

```bash
OPENAI_API_KEY=...
```

Optional stage-specific keys:

```bash
OPENAI_API_KEY_INTENT=...
OPENAI_API_KEY_PROFILE_SELECTION=...
OPENAI_API_KEY_PAGEINDEX=...
OPENAI_API_KEY_PROMPT_REFINEMENT=...
OPENAI_API_KEY_CONTENT_GENERATION=...
OPENAI_API_KEY_EVALUATION=...
```

Optional model overrides:

```bash
AURORA_AGENT_MODEL=gpt-4o
AURORA_CONTENT_MODEL=gpt-4o
AURORA_RETRIEVAL_MODEL=gpt-4o-mini
```

Optional audit log path:

```bash
AURORA_AUDIT_LOG_PATH=/app/logs/audit.jsonl
```

## Render

1. Create a new Web Service from the GitHub repo.
2. Set the root directory to `aurora_tool_server`.
3. Choose Docker as the runtime.
4. Add the secrets above.
5. Deploy.

Render provides `$PORT`; `deploy/start.sh` exposes Streamlit on that port and
runs FastAPI internally on `127.0.0.1:8000`.

## Railway

1. Create a new project from the GitHub repo.
2. Do not set a custom root directory.
3. Let Railway build from `Dockerfile.railway`.
4. Add the secrets above.
5. Generate a public domain. If it fails with "Application failed to respond",
   set the domain target port to the port shown in the Streamlit startup log.

## Security Note

The app currently has no login wall. A public URL can spend your OpenAI key and
expose the bundled context assets to anyone with the link. For anything beyond
a controlled demo, put it behind platform auth, Cloudflare Access, or another
authentication layer.
