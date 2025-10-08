#!/bin/bash

# Set default PORT if not provided
PORT=${PORT:-8080}

echo "=== Entrypoint Debug Info ==="
echo "PORT: ${PORT}"
echo "USER: $(whoami)"
echo "PWD: $(pwd)"
echo "Python version: $(python --version)"
echo "Checking app directory..."
ls -la /app/ || echo "Cannot list /app"
echo "Checking cache directory..."
ls -la /app/.cache/huggingface/ || echo "Cache dir doesn't exist"
echo "============================="

# Create cache directories (ignore errors if they exist)
mkdir -p /app/.cache/huggingface/transformers 2>/dev/null || true
mkdir -p /app/models 2>/dev/null || true

echo "Cache directories ready. Models will download on first use."
echo "Starting FastAPI server on port ${PORT}..."

# Test Python import before starting uvicorn
echo "Testing Python imports..."
python -c "import app.main" 2>&1 || {
    echo "ERROR: Failed to import app.main"
    echo "Trying to show the actual error:"
    python -c "import app.main"
    exit 1
}

echo "Python imports successful. Starting uvicorn..."

# Start FastAPI app 
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log
