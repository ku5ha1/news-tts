# Stage 1: Builder
FROM python:3.11 as builder

WORKDIR /app

# Install dependencies for the build stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime dependencies if needed (e.g., curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY app/ ./app/
COPY entrypoint.sh .

# Create non-root user and set permissions
RUN adduser --system --group app && \
    chown -R app:app /app && \
    chmod +x entrypoint.sh

USER app

# Create cache and model directories with proper ownership
RUN mkdir -p /app/.cache/huggingface/transformers && \
    mkdir -p /app/models

# Environment variables
ENV PORT=8080 \
    PYTHONUNBUFFERED=1


# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Start the application
ENTRYPOINT ["./entrypoint.sh"]