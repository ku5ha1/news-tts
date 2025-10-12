#!/bin/bash
set -ex

APP_USER="app"
HF_HOME="/root/.cache/huggingface"

# Use default HuggingFace cache location
LOG_LEVEL="${LOG_LEVEL:-info}"  # Default to lowercase 'info'

echo "Starting container as user: $(whoami)"
echo "LOG_LEVEL is set to: $LOG_LEVEL"
echo "HF_HOME is set to: $HF_HOME (default HuggingFace cache)"
echo "Environment validation:"
echo "  HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-not_set}"
echo "  TRUST_REMOTE_CODE: ${TRUST_REMOTE_CODE:-not_set}"
echo "  TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE:-not_set}"

if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root. Setting up default HuggingFace cache directory."

    # Create default HuggingFace cache directory if it doesn't exist
    if [ ! -d "$HF_HOME" ]; then
        echo "Creating default HuggingFace cache directory: $HF_HOME"
        mkdir -p "$HF_HOME"
    fi

    echo "Default HuggingFace cache directory ready: $HF_HOME"
    
    # Set ownership of the cache directory
    chown -R "$APP_USER":"$APP_USER" "$HF_HOME" || {
        echo "WARNING: Could not set ownership of $HF_HOME - continuing anyway"
    }
    echo "Ownership of $HF_HOME set to $APP_USER."

    # Verify app user can access the cache directory
    if ! su "$APP_USER" -c "test -r $HF_HOME"; then
        echo "WARNING: App user cannot access cache directory $HF_HOME - continuing anyway"
    else
        echo "Cache directory is accessible."
    fi

    # Set ownership of the entire app directory
    echo "Setting ownership of application files..."
    chown -R "$APP_USER":"$APP_USER" /app || {
        echo "WARNING: Could not set ownership of /app - continuing anyway"
    }
    chmod -R 755 /app || {
        echo "WARNING: Could not set permissions of /app - continuing anyway"
    }
    echo "Ownership of /app set to $APP_USER."

    # Debug: Check what's actually in /app
    echo "=== Debug: Contents of /app ==="
    ls -la /app/
    echo "=== Debug: Contents of /app/app ==="
    ls -la /app/app/ || echo "No /app/app directory found"
    echo "=== Debug: Check if main.py exists ==="
    ls -la /app/app/main.py || echo "main.py not found"
    echo "=== Debug: Check permissions ==="
    stat /app/app/main.py || echo "Cannot stat main.py"

    # Verify app user can access the app directory
    if ! su "$APP_USER" -c "test -r /app/app/main.py"; then
        echo "ERROR: App user cannot access application files!"
        echo "=== Debug: Trying to access as app user ==="
        su "$APP_USER" -c "ls -la /app/app/" || echo "Cannot list /app/app as app user"
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