# app/scripts/download_snapshots.py
import os
from huggingface_hub import snapshot_download

def main() -> None:
    
    cache_dir = os.getenv("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading dist-200M model snapshots into standard HF cache: {cache_dir}")

    revision = os.getenv("MODEL_REV") 

    print("Downloading IndicTrans2 EN→Indic dist-200M model...")
    snapshot_download(
        repo_id="ai4bharat/indictrans2-en-indic-dist-200M",
        revision=revision,
        cache_dir=cache_dir, 
        # local_dir_use_symlinks=False, # Not needed with cache_dir
        # local_files_only=True # Optional: ensure it doesn't try network if files exist (but they won't on first download)
    )

    print("Downloading IndicTrans2 Indic→EN dist-200M model...")
    snapshot_download(
        repo_id="ai4bharat/indictrans2-indic-en-dist-200M",
        revision=revision,
        cache_dir=cache_dir, 
        # local_dir_use_symlinks=False,
    )

    print("All translation models downloaded to standard cache (dist-200M only)")
    print("TTS will use ElevenLabs API - no local TTS models needed")

if __name__ == "__main__":
    main()