#!/bin/bash
set -e

APP_USER="app" # CORRECTED: Must match user created in Dockerfile
HF_HOME="/mnt/hf-cache" # CORRECT: Matches ACI MOUNT_PATH

echo "Starting container as user: $(whoami)"

# CRITICAL LOGIC: If we are root (which we should be via Dockerfile), fix permissions.
if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root. Granting $APP_USER access to mounted volume $HF_HOME."

    # Wait for mount to ensure chown doesn't fail
    while [ ! -d "$HF_HOME" ]; do
        echo "Waiting for $HF_HOME to mount..."
        sleep 1
    done

    # CRITICAL FIX: Change ownership of the mounted volume to the non-root user
    chown -R "$APP_USER":"$APP_USER" "$HF_HOME"
    echo "Ownership of $HF_HOME set to $APP_USER."

    # Switch user and run the application
    echo "Switching to user $APP_USER and starting uvicorn."
    # Use 'su -c' to switch to the non-root user 'app' before executing uvicorn
    exec su "$APP_USER" -c "uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level $LOG_LEVEL"
fi

# Fallback path (should not be reached if Dockerfile is set to USER root)
echo "Running as non-root user directly. Starting uvicorn."
exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"