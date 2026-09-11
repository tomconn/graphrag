#!/bin/sh
# Runs uvicorn in the background and the Vite dev server in the foreground.
# Forwards SIGTERM/SIGINT to both so `docker compose stop` shuts down cleanly.
set -e

uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir /app/backend &
BACKEND_PID=$!

VITE_PID=""
cleanup() {
    [ -n "$VITE_PID" ] && kill -TERM "$VITE_PID" 2>/dev/null || true
    kill -TERM "$BACKEND_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup TERM INT

cd /app/frontend
npm run dev -- --host 0.0.0.0 --port 3000 &
VITE_PID=$!

wait "$VITE_PID"