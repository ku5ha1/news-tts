#!/bin/bash

# Set default PORT if not provided
PORT=${PORT:-8080}

# Write debug output to Azure's log directory
DEBUG_LOG="/home/LogFiles/startup-debug.log"
mkdir -p /home/LogFiles 2>/dev/null || true

{
    echo "=== Entrypoint Debug Info $(date) ==="
    echo "PORT: ${PORT}"
    echo "USER: $(whoami)"
    echo "PWD: $(pwd)"
    echo "Python version: $(python --version 2>&1)"
    echo "Checking app directory..."
    ls -la /app/ 2>&1 || echo "Cannot list /app"
    echo "Checking cache directory..."
    ls -la /app/.cache/huggingface/ 2>&1 || echo "Cache dir doesn't exist"
    echo "============================="
    
    # Create cache directories (ignore errors if they exist)
    mkdir -p /app/.cache/huggingface/transformers 2>&1 || true
    mkdir -p /app/models 2>&1 || true
    
    echo "Cache directories ready. Models will download on first use."
    echo "Starting FastAPI server on port ${PORT}..."
    
    # Test Python import before starting uvicorn
    echo "Testing Python imports..."
    python -c "import sys; print('Python path:', sys.path)" 2>&1
    python -c "import app.main; print('Import successful!')" 2>&1 || {
        echo "ERROR: Failed to import app.main"
        echo "Detailed error:"
        python -c "import app.main" 2>&1
        exit 1
    }
    
    echo "Python imports successful. Starting uvicorn..."
} | tee -a "$DEBUG_LOG" 2>&1

# Start FastAPI app (also log to file)
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300 2>&1 | tee -a "$DEBUG_LOG"
