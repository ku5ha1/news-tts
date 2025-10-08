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

echo "Testing IndicTransToolkit import..." >&2
python -c "from IndicTransToolkit import IndicProcessor; print('IndicTransToolkit OK')" 2>&1 || {
    echo "ERROR: IndicTransToolkit import failed" >&2
    python -c "import IndicTransToolkit; print(dir(IndicTransToolkit))" 2>&1 || echo "Package not found" >&2
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
