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

# 👉 Pre-download AI4Bharat translation models during build (critical!)
# This avoids slow downloads when the app starts on Cloud Run
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN python -c <<EOF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('🔽 Downloading AI4Bharat IndicTrans2 models...')

AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True)
AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-en-indic-1B', trust_remote_code=True)

AutoTokenizer.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True)
AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/IndicTrans2-indic-en-1B', trust_remote_code=True)

print('✅ Models downloaded and saved to cache')
EOF

# Copy your app code into the container
COPY app/ ./app/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV AI4BHARAT_MODELS_PATH=/root/.cache/huggingface/transformers
ENV PYTHONUNBUFFERED=1

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Health check: confirm the app is responsive
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]