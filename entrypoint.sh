#!/bin/bash
set -x  # Enable debug mode

# Set default PORT if not provided
PORT=${PORT:-8080}

echo "=== ENTRYPOINT STARTING ===" >&2
echo "PORT: ${PORT}" >&2
echo "USER: $(whoami)" >&2

# -------------------------------
# Create cache directories at runtime
# -------------------------------
# For local cache
mkdir -p /app/.cache/huggingface || true

# For Azure File Share mount
if [ ! -d "/mnt/hf-cache" ]; then
    echo "WARNING: /mnt/hf-cache does not exist (Azure File Share not mounted?)" >&2
else
    echo "Creating subdirectories in /mnt/hf-cache..."
    mkdir -p /mnt/hf-cache/hub /mnt/hf-cache/transformers || true
    chmod -R 777 /mnt/hf-cache || true
fi

# -------------------------------
# Test Python and packages
# -------------------------------
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

# -------------------------------
# Start FastAPI
# -------------------------------
echo "Starting uvicorn..." >&2

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300
