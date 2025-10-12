# =========================
# Stage 1: Builder
# =========================
FROM python:3.11 AS builder
WORKDIR /app

# 1. Install build-time dependencies
# Pinning to python-dev is helpful for some C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies with cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# 3. Clone and install IndicTransToolkit from source (editable install)
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /tmp/IndicTransToolkit && \
    cd /tmp/IndicTransToolkit && \
    pip install --no-cache-dir --editable ./ && \
    cd /app && \
    rm -rf /tmp/IndicTransToolkit && \
    pip cache purge

# 4. Create models directory for runtime download
RUN mkdir -p /app/hf-cache && \
    echo "Models directory created - will download at runtime"

# 5. Clean up build dependencies to reduce image size
RUN apt-get autoremove -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim
WORKDIR /app

# 1. Install necessary runtime dependencies (curl, bash, and 'su' dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bash procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 2. Explicitly create the non-root 'app' user and group
# The user is consistently named 'app'
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

# 3. Copy files from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# 4. Clean up Python cache and temporary files
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/*

# 5. Set correct ownership for the 'app' user
RUN chown -R app:app /app

# 6. Copy and set executable permissions for the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 7. Final setup: Set user to root TEMPORARILY so the entrypoint can run chown
# The entrypoint will switch to the non-root user 'app' after fixing permissions.
USER root 
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    # Use runtime model download
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache \
    HF_HUB_OFFLINE=0 \
    TRUST_REMOTE_CODE=1

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]