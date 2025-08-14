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

# Copy Python dependencies file
COPY requirements.txt .

# Install Python packages (like FastAPI, transformers, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for model cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create cache directory
RUN mkdir -p /app/.cache/huggingface/transformers

# 👉 Pre-download AI4Bharat translation models during build (critical!)
# Download models in separate steps for better error handling and caching
RUN python -c "from transformers import AutoTokenizer; print('🔽 Downloading tokenizers...'); AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True, cache_dir='/app/.cache/huggingface/transformers'); print('✅ Downloaded en-indic tokenizer')"

RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True, cache_dir='/app/.cache/huggingface/transformers'); print('✅ Downloaded indic-en tokenizer')"

RUN python -c "from transformers import AutoModelForSeq2SeqLM; print('🔽 Downloading en-indic model...'); AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True, cache_dir='/app/.cache/huggingface/transformers'); print('✅ Downloaded en-indic model')"

RUN python -c "from transformers import AutoModelForSeq2SeqLM; print('🔽 Downloading indic-en model...'); AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True, cache_dir='/app/.cache/huggingface/transformers'); print('✅ Downloaded indic-en model')"

# Copy your app code into the container
COPY app/ ./app/

# Create a non-root user for security and give access to model cache
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Add a simple health check endpoint test
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the FastAPI app with uvicorn (use PORT env var from Cloud Run)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 300