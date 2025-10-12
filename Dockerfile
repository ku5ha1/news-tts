# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder
WORKDIR /app

# 1. Install build-time dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# 3. IndicTransToolkit will be installed via requirements.txt

# 4. Copy application source code
COPY app /app/app

# 5. Create models directory for runtime download
RUN mkdir -p /app/hf-cache && \
    echo "Models directory created - will download at runtime"

# 6. Clean up build dependencies to reduce image size
RUN apt-get autoremove -y build-essential git && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/* && \
    find /usr -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true


# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim
WORKDIR /app

# 1. Install necessary runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bash procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 2. Create the non-root 'app' user and group
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

# 3. Copy files from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# 4. Clean up Python cache
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/*

# 5. Set correct ownership for the 'app' user
RUN chown -R app:app /app && \
    chmod -R 755 /app

# 6. Copy and set executable permissions for the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 7. Set user to root (entrypoint will switch to 'app' user)
USER root 
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info \
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache \
    HF_HUB_OFFLINE=0 \
    TRUST_REMOTE_CODE=1

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]