# Use a slim Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed to build Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies file first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for model cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=${HF_HOME}/transformers
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Create cache directory
RUN mkdir -p ${HF_HOME}/transformers

# 👉 Pre-download models (consider smaller models if startup is too slow)
RUN python -c "from transformers import AutoTokenizer; print('🔽 Downloading tokenizers...'); AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); print('✅ Downloaded en-indic tokenizer')" && \
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); print('✅ Downloaded indic-en tokenizer')" && \
    python -c "from transformers import AutoModelForSeq2SeqLM; print('🔽 Downloading en-indic model...'); AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); print('✅ Downloaded en-indic model')" && \
    python -c "from transformers import AutoModelForSeq2SeqLM; print('🔽 Downloading indic-en model...'); AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); print('✅ Downloaded indic-en model')"

# Copy app code
COPY app/ ./app/

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check with longer timeout for model loading
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Optimized startup command
CMD exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log