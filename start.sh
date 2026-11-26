#!/usr/bin/env bash
set -e

# Start FastAPI backend in the background
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Ensure background API process is stopped if container exits
trap "kill $API_PID 2>/dev/null || true" EXIT INT TERM

# Start Streamlit frontend in the foreground
exec streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501