#!/bin/bash
set -x  # Enable debug mode to see what's executing

# Set default PORT if not provided
PORT=${PORT:-8080}

echo "=== ENTRYPOINT STARTING ===" >&2
echo "PORT: ${PORT}" >&2
echo "USER: $(whoami)" >&2

# Create cache directories
mkdir -p /app/.cache/huggingface/transformers || true
mkdir -p /app/models || true

echo "Testing Python..." >&2
python --version >&2

echo "Testing IndicTrans2 import..." >&2
python -c "import sys; sys.path.append('/app/IndicTrans2'); from IndicTrans2.inference.engine import Model; print('IndicTrans2 OK')" 2>&1 || {
    echo "ERROR: IndicTrans2 import failed" >&2
    python -c "import sys; sys.path.append('/app/IndicTrans2'); import IndicTrans2; print(dir(IndicTrans2))" 2>&1 || echo "Package not found" >&2
}

echo "Testing app.main import..." >&2
python -c "import app.main; print('app.main import OK')" 2>&1 || {
    echo "ERROR: Failed to import app.main" >&2
    python -c "import app.main" 2>&1
    exit 1
}

echo "Starting uvicorn..." >&2

# Start FastAPI app
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300
