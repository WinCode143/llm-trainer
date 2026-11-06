#!/usr/bin/env bash
# Starts LLM Trainer.
#   ./run.sh        → normal mode: builds the frontend if needed and serves everything on http://127.0.0.1:8000
#   ./run.sh dev    → development: backend with reload (8000) + Vite with HMR (http://localhost:5173)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d backend/.venv ]; then (cd backend && uv sync); fi
if [ ! -d frontend/node_modules ]; then (cd frontend && npm install); fi

if [ "${1:-}" = "dev" ]; then
  trap 'kill 0' EXIT
  (cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000 --reload --reload-dir app) &
  (cd frontend && npx vite) &
  wait
else
  if [ ! -d frontend/dist ] || [ -n "$(find frontend/src -newer frontend/dist -print -quit)" ]; then
    (cd frontend && npx vite build)
  fi
  cd backend && exec .venv/bin/python -m uvicorn app.main:app --host "${LLMT_HOST:-127.0.0.1}" --port "${LLMT_PORT:-8000}"
fi
