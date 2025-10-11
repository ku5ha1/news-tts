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

# 3. Clone and install IndicTransToolkit
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit
WORKDIR /app/IndicTransToolkit
RUN pip install --editable ./
WORKDIR /app
ENV PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

# REMOVED: Pre-downloading model here, as the model must be loaded from the mounted volume /mnt/hf-cache.


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
    # Point HF_HOME to the mounted volume path expected by ACI
    HF_HOME=/mnt/hf-cache \
    TRANSFORMERS_CACHE=/mnt/hf-cache \
    PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]