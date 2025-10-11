#!/bin/bash
set -ex 

APP_USER="app"
HF_HOME="/mnt/hf-cache"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "Starting container as user: $(whoami)"
echo "LOG_LEVEL is set to: $LOG_LEVEL"
echo "HF_HOME is set to: $HF_HOME"

if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root. Granting $APP_USER access to mounted volume $HF_HOME."

    # Wait for mount with timeout
    timeout=60
    counter=0
    while [ ! -d "$HF_HOME" ] && [ $counter -lt $timeout ]; do
        echo "Waiting for $HF_HOME to mount... ($counter/$timeout)"
        sleep 1
        counter=$((counter + 1))
    done

    if [ ! -d "$HF_HOME" ]; then
        echo "ERROR: $HF_HOME did not mount within $timeout seconds!"
        exit 1
    fi

    echo "Mount found. Setting ownership..."
    chown -R "$APP_USER":"$APP_USER" "$HF_HOME"
    echo "Ownership of $HF_HOME set to $APP_USER."

    # Verify app user can access the app directory
    if ! su "$APP_USER" -c "test -r /app/app/main.py"; then
        echo "ERROR: App user cannot access application files!"
        exit 1
    fi

    echo "Switching to user $APP_USER and starting uvicorn..."
    exec su "$APP_USER" -c "cd /app && PYTHONPATH=/app/IndicTransToolkit:\$PYTHONPATH uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level $LOG_LEVEL"

fi

echo "Running as non-root user directly. Starting uvicorn."
# Check if /app exists (container) or use current directory (local)
if [ -d "/app" ]; then
    echo "Using container directory /app"
    cd /app
    exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"
else
    echo "Using current directory for local testing"
    exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"
fi