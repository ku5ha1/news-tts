# =========================
# Single Stage Build (Optimized)
# =========================
FROM python:3.11-slim
WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        bash \
        procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 2. Create the non-root 'app' user and group
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

# 3. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# 4. Copy application source code
COPY app /app/app

# 5. Create app user's home directory and cache directory
RUN mkdir -p /home/app/.cache/huggingface && \
    echo "App user's HuggingFace cache directory created - will download at runtime"

# 6. Clean up build dependencies to reduce image size
RUN apt-get autoremove -y build-essential git && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 7. Set correct ownership for the 'app' user
RUN chown -R app:app /app && \
    chmod -R 755 /app

# 8. Copy and set executable permissions for the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 9. Set user to root (entrypoint will switch to 'app' user)
USER root 
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    HF_HUB_OFFLINE=0 \
    TRUST_REMOTE_CODE=1

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]