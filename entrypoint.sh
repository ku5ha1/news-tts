# =========================
# Stage 1: Builder
# =========================
FROM python:3.11 AS builder
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone IndicTransToolkit and install
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit
WORKDIR /app/IndicTransToolkit
RUN pip install --editable ./
WORKDIR /app
ENV PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

# Pre-download HF model to /app/hf-cache
RUN mkdir -p /app/hf-cache && \
    python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', cache_dir='/app/hf-cache', trust_remote_code=True); \
    AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', cache_dir='/app/hf-cache', trust_remote_code=True)"
    
# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

USER app
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_CACHE=/app/hf-cache \
    PYTHONPATH="/app/IndicTransToolkit:${PYTHONPATH}"

EXPOSE ${PORT}
ENTRYPOINT ["./entrypoint.sh"]
