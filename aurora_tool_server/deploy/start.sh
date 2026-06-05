#!/usr/bin/env sh
set -eu

api_host="${AURORA_API_HOST:-127.0.0.1}"
api_port="${AURORA_API_PORT:-8000}"
streamlit_host="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
streamlit_port="${PORT:-8501}"

echo "Starting AURORA API on $api_host:$api_port"
uv run uvicorn aurora_tool_server.api:app --host "$api_host" --port "$api_port" &
api_pid="$!"

cleanup() {
    kill "$api_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if curl -fsS "http://$api_host:$api_port/health" >/dev/null; then
        echo "AURORA API is healthy"
        break
    fi
    if ! kill -0 "$api_pid" 2>/dev/null; then
        echo "AURORA API exited before becoming healthy" >&2
        wait "$api_pid"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://$api_host:$api_port/health" >/dev/null; then
    echo "AURORA API did not become healthy at http://$api_host:$api_port/health" >&2
    exit 1
fi

echo "Starting Streamlit on $streamlit_host:$streamlit_port"
uv run streamlit run frontend/app.py \
    --server.address="$streamlit_host" \
    --server.port="$streamlit_port" \
    --server.headless=true
