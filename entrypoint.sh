#!/bin/bash
set -e

# ===============================================
# 1. Initialization and Environment Checks
# ===============================================

echo "Starting container as user: $(whoami)"

# Check if the mandatory Hugging Face cache directory exists and is accessible.
if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: Hugging Face cache directory ($HF_HOME) not found. The model files may be missing."
    exit 1
fi

# ===============================================
# 2. Application Startup
# ===============================================

echo "Starting the IndicTrans2 service on port $PORT..."

# Replace the following line with the actual command to start your Python server.
# Common examples include: gunicorn, uvicorn, or a direct python command.

# Example 1: Starting a server using Uvicorn (common for FastAPI/Starlette)
# This assumes your application entry point is 'main:app' and you installed uvicorn/gunicorn.
# exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4

# Example 2: Starting a server using Gunicorn (common for Flask/Django)
# This assumes you have a Gunicorn configuration file or simple entry point.
# exec gunicorn --bind 0.0.0.0:$PORT --workers 4 'main:app'

# Example 3: Running a simple Python script
# If your application is a simple Python file named 'run_app.py'
exec python run_app.py --port $PORT

# Note: Using 'exec' replaces the current shell process with the application process.
# This ensures signals (like SIGTERM) are correctly handled by the Python application,
# which is crucial for graceful shutdown in Kubernetes or Docker Swarm.

# ===============================================