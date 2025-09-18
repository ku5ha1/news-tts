FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (audio libs removed since ElevenLabs API is used)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (leverages Docker cache)
COPY requirements.txt .

# Install Python packages with more verbose output for debugging
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Env vars
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Create cache & temp directories with proper permissions
RUN mkdir -p ${HF_HOME}/transformers /tmp /var/tmp /usr/tmp /app/tmp && \
    chmod 1777 /tmp /var/tmp /usr/tmp && \
    chmod 755 /app/tmp

# Copy application code
COPY app/ ./app/

# Download dist-200M models during build (baked into image)
# Use a heredoc to avoid quoting issues in Docker RUN
RUN python3 - <<'PY'
from huggingface_hub import snapshot_download
print('Downloading dist-200M translation models...')
snapshot_download(
    repo_id='ai4bharat/indictrans2-en-indic-dist-200M',
    local_dir='/app/models/indictrans2-en-indic-dist-200M',
    local_dir_use_symlinks=False,
    cache_dir='/app/.cache/huggingface'
)
snapshot_download(
    repo_id='ai4bharat/indictrans2-indic-en-dist-200M',
    local_dir='/app/models/indictrans2-indic-en-dist-200M',
    local_dir_use_symlinks=False,
    cache_dir='/app/.cache/huggingface'
)
print('Translation models downloaded successfully')
print('TTS will use ElevenLabs API - no local models needed')
PY

# Copy entrypoint and make executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user and give ownership of /app
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

# Health check (curl is installed above)
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

ENTRYPOINT ["/app/entrypoint.sh"]
