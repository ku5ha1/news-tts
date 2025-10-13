# =========================
# Multi-Stage Build with Model Preloading
# =========================

# Stage 1: Model Download and Cache
FROM python:3.11-slim as model-cache
WORKDIR /app

# Install system dependencies for model download
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model download
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch>=2.5.0 \
    transformers>=4.51.0 \
    huggingface_hub>=0.15.0 \
    sentencepiece==0.2.0 \
    sacremoses==0.1.1 \
    protobuf==4.25.3 \
    indictranstoolkit>=1.1.0

# Create cache directory and download models
RUN mkdir -p /models/cache && \
    export HF_HOME=/models/cache && \
    export TRANSFORMERS_CACHE=/models/cache && \
    python -c "
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indictranstoolkit import IndicTransToolkit

print('Downloading IndicTrans2 models...')
# Download EN->Indic model
en_indic_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', trust_remote_code=True)
en_indic_model = AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', trust_remote_code=True)

# Download Indic->EN model  
indic_en_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indictrans2-indic-en-dist-200M', trust_remote_code=True)
indic_en_model = AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-indic-en-dist-200M', trust_remote_code=True)

print('Models downloaded successfully!')
"

# Stage 2: Production Image
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        bash \
        procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create the non-root 'app' user and group
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Copy application source code
COPY app /app/app

# Copy pre-downloaded models from model-cache stage
COPY --from=model-cache /models/cache /home/app/.cache/huggingface

# Create app user's home directory
RUN mkdir -p /home/app && \
    echo "Pre-loaded models available in HuggingFace cache"

# Clean up build dependencies to reduce image size
RUN apt-get autoremove -y build-essential git && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set correct ownership for the 'app' user
RUN chown -R app:app /app /home/app && \
    chmod -R 755 /app /home/app

# Copy and set executable permissions for the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set user to root (entrypoint will switch to 'app' user)
USER root
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRUST_REMOTE_CODE=1

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]