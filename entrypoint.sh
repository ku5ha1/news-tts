#!/bin/bash
set -ex 

APP_USER="app"
HF_HOME="/mnt/hf-cache"

echo "Starting container as user: $(whoami)"

if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root. Granting $APP_USER access to mounted volume $HF_HOME."

    while [ ! -d "$HF_HOME" ]; do
        echo "Waiting for $HF_HOME to mount..."
        sleep 1
    done

    chown -R "$APP_USER":"$APP_USER" "$HF_HOME"
    echo "Ownership of $HF_HOME set to $APP_USER."

    echo "Switching to user $APP_USER and starting uvicorn with explicit path."

    exec su "$APP_USER" -c "PYTHONPATH=/app/IndicTransToolkit:\$PYTHONPATH uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level $LOG_LEVEL"

fi

echo "Running as non-root user directly. Starting uvicorn."
exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"