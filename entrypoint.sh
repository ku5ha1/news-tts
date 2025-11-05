#!/bin/bash
set -ex

APP_USER="app"
HF_HOME="/home/app/.cache/huggingface"

# Use app user's home directory for cache
LOG_LEVEL="${LOG_LEVEL:-info}"  # Default to lowercase 'info'

echo "Starting container as user: $(whoami)"
echo "LOG_LEVEL is set to: $LOG_LEVEL"
echo "HF_HOME is set to: $HF_HOME (pre-loaded models)"
echo "Environment validation:"
echo "  HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-not_set}"
echo "  TRUST_REMOTE_CODE: ${TRUST_REMOTE_CODE:-not_set}"
echo "  TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE:-not_set}"
echo "  AZURE_SPEECH_KEY: ${AZURE_SPEECH_KEY:-not_set}"
echo "  AZURE_SPEECH_REGION: ${AZURE_SPEECH_REGION:-not_set}"
echo "  AZURE_SPEECH_ENDPOINT: ${AZURE_SPEECH_ENDPOINT:-not_set}"

if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root. Setting up pre-loaded models for app user."

    # Verify pre-loaded models exist
    if [ -d "$HF_HOME" ] && [ "$(ls -A $HF_HOME)" ]; then
        echo "✅ Pre-loaded models found in $HF_HOME"
        echo "Model files:"
        ls -la "$HF_HOME" | head -10
    else
        echo "⚠️  No pre-loaded models found in $HF_HOME"
        echo "Models will be downloaded at runtime (slower startup)"
    fi

    # Create app user's home directory and cache directory
    mkdir -p "/home/app"
    if [ ! -d "$HF_HOME" ]; then
        echo "Creating HuggingFace cache directory: $HF_HOME"
        mkdir -p "$HF_HOME"
    fi

    echo "HuggingFace cache directory ready: $HF_HOME"
    
    # Set ownership and permissions of the cache directory
    chown -R "$APP_USER":"$APP_USER" "$HF_HOME" || {
        echo "WARNING: Could not set ownership of $HF_HOME - continuing anyway"
    }
    chmod -R 755 "$HF_HOME" || {
        echo "WARNING: Could not set permissions of $HF_HOME - continuing anyway"
    }
    echo "Ownership and permissions of $HF_HOME set to $APP_USER."

    # Verify app user can access and write to the cache directory
    if ! su "$APP_USER" -c "test -w $HF_HOME"; then
        echo "ERROR: App user cannot write to cache directory $HF_HOME"
        echo "=== Debug: Cache directory permissions ==="
        ls -la "$HF_HOME" || echo "Cannot list cache directory"
        echo "=== Debug: Trying to create test file as app user ==="
        su "$APP_USER" -c "touch $HF_HOME/test_write.tmp && rm $HF_HOME/test_write.tmp" || echo "Cannot write test file"
        exit 1
    else
        echo "Cache directory is writable by app user."
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

    # Check SSL certificates as root (before switching users)
    SSL_CERT_PATH="/etc/letsencrypt/live/diprkarnataka.duckdns.org/fullchain.pem"
    SSL_KEY_PATH="/etc/letsencrypt/live/diprkarnataka.duckdns.org/privkey.pem"
    
    if [ -f "$SSL_CERT_PATH" ] && [ -f "$SSL_KEY_PATH" ]; then
        echo "✅ SSL certificates found at: $SSL_CERT_PATH"
        # Make certs readable by app user (if needed)
        # Cert files should be readable, key should be readable by owner only
        chmod 644 "$SSL_CERT_PATH" 2>/dev/null || true
        chmod 644 "$SSL_KEY_PATH" 2>/dev/null || true
        # Try to make them readable by app user group (if in same group)
        chgrp "$APP_USER" "$SSL_CERT_PATH" 2>/dev/null || true
        chgrp "$APP_USER" "$SSL_KEY_PATH" 2>/dev/null || true
        export SSL_AVAILABLE="true"
        echo "SSL certificates configured for app user"
    else
        echo "⚠️  SSL certificates not found at: $SSL_CERT_PATH"
        echo "Will use HTTP server on port 8080"
        export SSL_AVAILABLE="false"
    fi

    echo "Switching to user $APP_USER and starting uvicorn..."
    echo "About to execute: su $APP_USER -c 'cd /app && python -m app.main'"
    # Pass SSL_AVAILABLE environment variable to the app user
    exec su "$APP_USER" -c "cd /app && export SSL_AVAILABLE=\"$SSL_AVAILABLE\" && echo 'Starting uvicorn with SSL support...' && python -m app.main"

fi

echo "Running as non-root user directly. Starting uvicorn."
# Check if /app exists (container) or use current directory (local)
if [ -d "/app" ]; then
    echo "Using container directory /app"
    cd /app
    exec python -m app.main
else
    echo "Using current directory for local testing"
    exec python -m app.main
fi