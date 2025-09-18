#!/bin/bash
set -e

# Download models if not already present
if [ ! -d "/app/models/indictrans2-en-indic-dist-200M" ]; then
    echo "Downloading EN->Indic dist-200M..."
    python3 -m huggingface_hub.snapshot_download \
        ai4bharat/indictrans2-en-indic-dist-200M \
        --local-dir /app/models/indictrans2-en-indic-dist-200M \
        --local-dir-use-symlinks False
fi

if [ ! -d "/app/models/indictrans2-indic-en-dist-200M" ]; then
    echo "Downloading Indic->EN dist-200M..."
    python3 -m huggingface_hub.snapshot_download \
        ai4bharat/indictrans2-indic-en-dist-200M \
        --local-dir /app/models/indictrans2-indic-en-dist-200M \
        --local-dir-use-symlinks False
fi

echo "Models ready. Starting FastAPI server..."

# Start FastAPI app
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log
