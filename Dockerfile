# =========================
# Stage 1: Builder
# =========================
# Fix to force new build
FROM python:3.11 AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone IndicTransToolkit repository and install
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit
WORKDIR /app/IndicTransToolkit
RUN pip install --editable ./
WORKDIR /app


# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/IndicTransToolkit /app/IndicTransToolkit

# Install runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code and scripts
COPY app/ ./app/
COPY entrypoint.sh .

# IndicTransToolkit handles model loading automatically

# Setup user + permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app && \
    chmod +x entrypoint.sh

USER app

# Env vars
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    TRUST_REMOTE_CODE=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ready || exit 1

EXPOSE ${PORT}

RUN chmod 755 /app/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]