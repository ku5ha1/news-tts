# =========================
# Stage 1: Builder
# =========================
# Fix to force new build
FROM python:3.11 as builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone IndicTrans2 repository manually
RUN git clone https://github.com/AI4Bharat/IndicTrans2.git /app/IndicTrans2

# Copy scripts
COPY app/scripts/download_snapshots.py app/scripts/preload_models.py /app/scripts/

# Download translation models
RUN python /app/scripts/download_snapshots.py

# Preload models into cache (ensures they can be loaded)
RUN python /app/scripts/preload_models.py


# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and cached models from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface
COPY --from=builder /app/IndicTrans2 /app/IndicTrans2

# Install runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code and scripts
COPY app/ ./app/
COPY entrypoint.sh .

# Copy verification script
COPY app/scripts/verify_cache.py /app/scripts/

# Optional: quick cache verification (non-blocking)
RUN python /app/scripts/verify_cache.py || true

# Setup user + permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app && \
    chmod +x entrypoint.sh

USER app

# Env vars
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_HUB_OFFLINE=1 \
    TRUST_REMOTE_CODE=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ready || exit 1

EXPOSE ${PORT}

RUN chmod 755 /app/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]