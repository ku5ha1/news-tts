# =========================
# Stage 1: Builder
# =========================
# Force new build
FROM python:3.11 as builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and services
COPY app/scripts/download_snapshots.py app/scripts/preload_models.py /app/scripts/
COPY app/services/translation_service.py /app/services/

# Download translation models (snapshots)
RUN python /app/scripts/download_snapshots.py

# Preload models into cache (ensures they can be loaded)
RUN python /app/scripts/preload_models.py

# Force warmup of TranslationService (loads both EN→Indic and Indic→EN models)
RUN python -c "from app.services.translation_service import translation_service; translation_service.warmup()"

# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim

WORKDIR /app

# Copy application code and scripts
COPY app/ ./app/
COPY entrypoint.sh .

# Copy verification script
COPY app/scripts/verify_cache.py /app/scripts/

# Warmup translation service (optional but recommended)
RUN python -c "from app.services.translation_service import translation_service; translation_service.warmup()" || true

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

ENTRYPOINT ["./entrypoint.sh"]
