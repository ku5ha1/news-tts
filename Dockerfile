FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git curl libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first
COPY requirements.txt .

# Install Python packages with more verbose output for debugging
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Env vars
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=${HF_HOME}/transformers
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Create cache directories
RUN mkdir -p ${HF_HOME}/transformers

# Copy app code
COPY app/ ./app/

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user and set permissions
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

EXPOSE ${PORT}

ENTRYPOINT ["/app/entrypoint.sh"]