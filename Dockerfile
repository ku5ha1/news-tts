# Stage 1: Builder
FROM python:3.11 as builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models with explicit cache directory and verification
RUN python -c " \
    import os; \
    import sys; \
    import json; \
    # Set environment variables for Hugging Face cache \
    os.environ['HF_HOME'] = '/app/.cache/huggingface'; \
    os.environ['HF_HUB_CACHE'] = '/app/.cache/huggingface/hub'; \
    os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/huggingface/transformers'; \
    # CRITICAL: Trust remote code for IndicTrans2 models \
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'; \
    cache_dir = os.environ['HF_HUB_CACHE']; \
    print(f'Using cache directory: {cache_dir}'); \
    \
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig; \
    en_indic_model='ai4bharat/indictrans2-en-indic-dist-200M'; \
    indic_en_model='ai4bharat/indictrans2-indic-en-dist-200M'; \
    \
    print('--- Downloading and verifying EN->Indic model ---'); \
    # Download config first to verify it has model_type \
    config_en = AutoConfig.from_pretrained(en_indic_model, trust_remote_code=True, cache_dir=cache_dir); \
    print(f'EN->Indic model type: {getattr(config_en, \"model_type\", \"MISSING\")}'); \
    tokenizer_en = AutoTokenizer.from_pretrained(en_indic_model, trust_remote_code=True, cache_dir=cache_dir); \
    model_en = AutoModelForSeq2SeqLM.from_pretrained(en_indic_model, trust_remote_code=True, cache_dir=cache_dir); \
    print('EN->Indic model downloaded and loaded successfully.'); \
    \
    print('--- Downloading and verifying Indic->EN model ---'); \
    config_in = AutoConfig.from_pretrained(indic_en_model, trust_remote_code=True, cache_dir=cache_dir); \
    print(f'Indic->EN model type: {getattr(config_in, \"model_type\", \"MISSING\")}'); \
    tokenizer_in = AutoTokenizer.from_pretrained(indic_en_model, trust_remote_code=True, cache_dir=cache_dir); \
    model_in = AutoModelForSeq2SeqLM.from_pretrained(indic_en_model, trust_remote_code=True, cache_dir=cache_dir); \
    print('Indic->EN model downloaded and loaded successfully.'); \
    \
    # Test a simple translation to ensure models work \
    print('--- Testing models functionality ---'); \
    # Simple test with EN->Indic \
    test_input = tokenizer_en.encode('Hello world', return_tensors='pt'); \
    with model_en.eval(): \
        test_output = model_en.generate(test_input, max_length=50, num_beams=1, do_sample=False); \
    print('EN->Indic test generation successful'); \
    \
    # Test with Indic->EN \
    test_input_in = tokenizer_in.encode('नमस्ते', return_tensors='pt'); \
    with model_in.eval(): \
        test_output_in = model_in.generate(test_input_in, max_length=50, num_beams=1, do_sample=False); \
    print('Indic->EN test generation successful'); \
    \
    # Verification of cache structure \
    print('--- Verifying downloaded models in cache ---'); \
    from huggingface_hub import scan_cache_dir; \
    hf_cache_info = scan_cache_dir(cache_dir); \
    print(f'Cache repos found: {[repo.repo_id for repo in hf_cache_info.repos]}'); \
    \
    expected_repos = ['ai4bharat/indictrans2-en-indic-dist-200M', 'ai4bharat/indictrans2-indic-en-dist-200M']; \
    cached_repos = [repo.repo_id for repo in hf_cache_info.repos]; \
    for repo_id in expected_repos: \
        if repo_id not in cached_repos: \
            print(f'ERROR: Expected repo {repo_id} not found in cache!', file=sys.stderr); \
            sys.exit(1); \
        else: \
            repo = next(r for r in hf_cache_info.repos if r.repo_id == repo_id); \
            print(f'Found repo: {repo.repo_id}'); \
            if not repo.revisions: \
                print(f'ERROR: No revisions found for {repo_id}!', file=sys.stderr); \
                sys.exit(1); \
            latest_rev = list(repo.revisions)[0]; \
            print(f'  Latest revision: {latest_rev.commit_hash}'); \
            \
            # Verify essential files exist \
            essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']; \
            for essential_file in essential_files: \
                file_path = os.path.join(latest_rev.snapshot_path, essential_file); \
                if os.path.exists(file_path): \
                    print(f'  Found: {essential_file}'); \
                else: \
                    print(f'  Warning: {essential_file} not found'); \
            \
            # Check for model files \
            model_files_found = False; \
            potential_model_files = ['pytorch_model.bin', 'model.safetensors']; \
            for model_file in potential_model_files: \
                model_file_path = os.path.join(latest_rev.snapshot_path, model_file); \
                if os.path.exists(model_file_path): \
                    model_files_found = True; \
                    print(f'  Found model file: {model_file}'); \
                    break; \
            \
            # Check for sharded models \
            if not model_files_found: \
                for potential_index in ['pytorch_model.bin.index.json', 'model.safetensors.index.json']; \
                    index_path = os.path.join(latest_rev.snapshot_path, potential_index); \
                    if os.path.exists(index_path): \
                        model_files_found = True; \
                        print(f'  Found sharded model index: {potential_index}'); \
                        break; \
            \
            if not model_files_found: \
                print(f'ERROR: No model files found for {repo_id}!', file=sys.stderr); \
                sys.exit(1); \
    \
    print('--- All models downloaded, verified, and tested successfully ---'); \
    "

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and cached models
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/
COPY entrypoint.sh .

# Set proper permissions and create directories
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app && \
    chmod +x entrypoint.sh

# Verify models are accessible in production stage
RUN python -c " \
    import os; \
    cache_dir = '/app/.cache/huggingface/hub'; \
    print(f'Verifying cache in production stage: {cache_dir}'); \
    if not os.path.exists(cache_dir): \
        print('ERROR: Cache directory not found in production stage!'); \
        exit(1); \
    from huggingface_hub import scan_cache_dir; \
    hf_cache_info = scan_cache_dir(cache_dir); \
    expected_repos = ['ai4bharat/indictrans2-en-indic-dist-200M', 'ai4bharat/indictrans2-indic-en-dist-200M']; \
    cached_repos = [repo.repo_id for repo in hf_cache_info.repos]; \
    for repo_id in expected_repos: \
        if repo_id not in cached_repos: \
            print(f'ERROR: {repo_id} not found in production cache!'); \
            exit(1); \
        else: \
            # Verify config.json exists and has proper content \
            repo = next(r for r in hf_cache_info.repos if r.repo_id == repo_id); \
            latest_rev = list(repo.revisions)[0]; \
            config_path = os.path.join(latest_rev.snapshot_path, 'config.json'); \
            if os.path.exists(config_path): \
                import json; \
                with open(config_path, 'r') as f: \
                    config_data = json.load(f); \
                print(f'  Config for {repo_id}: model_type = {config_data.get(\"model_type\", \"MISSING\")}'); \
            else: \
                print(f'ERROR: config.json not found for {repo_id}!'); \
                exit(1); \
    print('Production stage model verification successful!'); \
    "

USER app

# Environment variables - CRITICAL: Add trust_remote_code setting
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_HUB_OFFLINE=1 \
    TRUST_REMOTE_CODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ready || exit 1

EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]