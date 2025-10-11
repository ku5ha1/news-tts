# =========================
# Stage 1: Builder
# =========================
FROM python:3.11 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone IndicTransToolkit and install
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit
WORKDIR /app/IndicTransToolkit
RUN pip install --editable ./
WORKDIR /app

# Add IndicTransToolkit to Python path
ENV PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/IndicTransToolkit /app/IndicTransToolkit

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code and entrypoint
COPY app/ ./app/
COPY entrypoint.sh .

# Setup user, cache directory, and permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /app/.cache/huggingface && \
    chown -R app:app /app && \
    chmod +x entrypoint.sh

USER app

# Environment variables
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    TRUST_REMOTE_CODE=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]
