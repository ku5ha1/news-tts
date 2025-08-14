# Use a slim Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
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

# Pre-download AI4Bharat models
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); \
    AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); \
    AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}'); \
    AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True, cache_dir='${TRANSFORMERS_CACHE}')" 

# Copy application code
COPY app/ ./app/

# Add warm-up script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
