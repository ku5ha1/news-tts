# =========================
# Stage 1: Builder
# =========================
FROM python:3.11 AS builder
WORKDIR /app

# 1. Install build-time dependencies
# Pinning to python-dev is helpful for some C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Install IndicTransToolkit from PyPI (no compilation needed)
RUN pip install IndicTransToolkit

# 4. Pre-download models to bake them into the image
RUN python -c "import os; os.environ['HF_HOME'] = '/app/hf-cache'; os.environ['TRANSFORMERS_CACHE'] = '/app/hf-cache'; os.environ['HF_HUB_OFFLINE'] = '0'; from IndicTransToolkit import IndicProcessor; processor = IndicProcessor()"


# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim
WORKDIR /app

# 1. Install necessary runtime dependencies (curl, bash, and 'su' dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bash procps && \
    rm -rf /var/lib/apt/lists/*

# 2. Explicitly create the non-root 'app' user and group
# The user is consistently named 'app'
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

# 3. Copy files from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# 4. Set correct ownership for the 'app' user
RUN chown -R app:app /app

# 5. Copy and set executable permissions for the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 6. Final setup: Set user to root TEMPORARILY so the entrypoint can run chown
# The entrypoint will switch to the non-root user 'app' after fixing permissions.
USER root 
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    # Use baked-in models (no file share needed)
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRUST_REMOTE_CODE=1

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]