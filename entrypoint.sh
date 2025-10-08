#!/bin/bash
set -e

# Set default PORT if not provided
PORT=${PORT:-8080}

# Create cache directories (ignore errors if they exist)
mkdir -p /app/.cache/huggingface/transformers 2>/dev/null || true
mkdir -p /app/models 2>/dev/null || true

echo "Cache directories ready. Models will download on first use."
echo "Starting FastAPI server on port ${PORT}..."

# Start FastAPI app 
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log
