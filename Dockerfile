# =========================
# Single-Stage Build (No Model Preloading)
# =========================

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

# Create app user's home directory
RUN mkdir -p /home/app && \
    echo "Models will be downloaded at runtime"

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
    HF_HUB_OFFLINE=0 \
    TRUST_REMOTE_CODE=1

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]