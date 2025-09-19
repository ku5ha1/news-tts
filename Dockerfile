# Stage 1: Builder
FROM python:3.11 as builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    cache_dir='/app/.cache/huggingface/transformers'; \
    en_indic_model='ai4bharat/indictrans2-en-indic-dist-200M'; \
    indic_en_model='ai4bharat/indictrans2-indic-en-dist-200M'; \
    AutoTokenizer.from_pretrained(en_indic_model, trust_remote_code=True, cache_dir=cache_dir); \
    AutoModelForSeq2SeqLM.from_pretrained(en_indic_model, trust_remote_code=True, cache_dir=cache_dir); \
    AutoTokenizer.from_pretrained(indic_en_model, trust_remote_code=True, cache_dir=cache_dir); \
    AutoModelForSeq2SeqLM.from_pretrained(indic_en_model, trust_remote_code=True, cache_dir=cache_dir);"

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and cached models
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/
COPY entrypoint.sh .

# Set proper permissions
RUN adduser --system --group app && \
    chown -R app:app /app && \
    chmod +x entrypoint.sh

USER app

# Environment variables
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ready || exit 1

EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]