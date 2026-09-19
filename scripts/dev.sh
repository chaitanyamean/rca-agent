#!/usr/bin/env bash
# dev.sh — convenience script to start the development server
set -euo pipefail

echo "Starting rca-agent in development mode..."
uvicorn rca_agent.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --reload \
  --log-level debug
