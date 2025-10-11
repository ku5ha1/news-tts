# =========================
# Stage 1: Builder
# =========================
# Use a full Python image for the build stage to ensure all dependencies are available
FROM python:3.11 AS builder
WORKDIR /app

# 1. Install build-time dependencies
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

# 4. Pre-download HF model to /app/hf-cache
# This is crucial for avoiding a slow startup and internet dependency later.
RUN mkdir -p /app/hf-cache && \
    python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', cache_dir='/app/hf-cache', trust_remote_code=True); \
    AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', cache_dir='/app/hf-cache', trust_remote_code=True)"


# =========================
# Stage 2: Production
# =========================
# Use the minimal slim image for a smaller final container
FROM python:3.11-slim
WORKDIR /app

# 1. Install necessary runtime dependencies (curl and bash)
# 'bash' is added as entrypoint.sh scripts often rely on it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bash && \
    rm -rf /var/lib/apt/lists/*

# 2. Explicitly create the non-root 'app' user and group
# This is the most likely fix for your 'Terminated' error.
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

# 3. Copy files from the builder stage
# /usr/local/lib/python3.11/site-packages contains all installed Python packages.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# /usr/local/bin contains executables (like the installed Python scripts).
COPY --from=builder /usr/local/bin /usr/local/bin
# /app contains your source code, the cloned repo, and the /app/hf-cache.
COPY --from=builder /app /app

# 4. Set correct ownership for the 'app' user
# This ensures the non-root user can read/write to the application and cache directories.
RUN chown -R app:app /app

# 5. Copy and set executable permissions for the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 6. Final setup
USER app
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    # Set the cache variables again, pointing to the location we copied
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache \
    PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]