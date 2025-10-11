#!/bin/bash
set -e

# ===============================================
# 1. Initialization and Environment Checks
# ===============================================

echo "Starting container as user: $(whoami)"

if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: Hugging Face cache directory ($HF_HOME) not found. The model files may be missing."
    exit 1
fi

echo "Starting the IndicTrans2 service on port $PORT..."

exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"