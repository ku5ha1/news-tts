#!/bin/bash
set -ex 

APP_USER="app"
HF_HOME="/app/hf-cache"  # Use baked-in models
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "Starting container as user: $(whoami)"
echo "LOG_LEVEL is set to: $LOG_LEVEL"
echo "HF_HOME is set to: $HF_HOME (baked-in models)"
echo "Environment validation:"
echo "  HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-not_set}"
echo "  TRUST_REMOTE_CODE: ${TRUST_REMOTE_CODE:-not_set}"
echo "  TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE:-not_set}"

if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root. Setting up baked-in models directory."

    # Verify baked-in models directory exists and is accessible
    if [ ! -d "$HF_HOME" ]; then
        echo "ERROR: Baked-in models directory $HF_HOME does not exist!"
        echo "This means models were not properly baked into the image during build."
        exit 1
    fi

    echo "Baked-in models directory found: $HF_HOME"
    
    # Set ownership of the models directory
    chown -R "$APP_USER":"$APP_USER" "$HF_HOME" || {
        echo "WARNING: Could not set ownership of $HF_HOME - continuing anyway"
    }
    echo "Ownership of $HF_HOME set to $APP_USER."

    # Verify app user can access the models
    if ! su "$APP_USER" -c "test -r $HF_HOME"; then
        echo "WARNING: App user cannot access models directory $HF_HOME - continuing anyway"
    else
        echo "Models directory is accessible."
    fi

    # Verify app user can access the app directory
    if ! su "$APP_USER" -c "test -r /app/app/main.py"; then
        echo "ERROR: App user cannot access application files!"
        exit 1
    fi
    echo "Application files are accessible."

    echo "Switching to user $APP_USER and starting uvicorn..."
    echo "About to execute: su $APP_USER -c 'cd /app && uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level $LOG_LEVEL'"
    exec su "$APP_USER" -c "cd /app && echo 'Starting uvicorn...' && uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level $LOG_LEVEL"

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