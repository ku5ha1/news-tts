#!/bin/bash
set -e

# Create cache directories
mkdir -p /app/.cache/huggingface/transformers
mkdir -p /app/models

echo "Cache directories ready. Models will download on first use."
echo "Starting FastAPI server..."

# Start FastAPI app 
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log
