# app/scripts/download_snapshots.py
import os
from huggingface_hub import snapshot_download

def main() -> None:
    
    cache_dir = os.getenv("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading dist-200M model snapshots into standard HF cache: {cache_dir}")

    revision = os.getenv("MODEL_REV") 
    print(f"Using revision: {revision if revision else 'main (default)'}")

    try:
        print("Downloading IndicTrans2 EN→Indic dist-200M model...")
        en_indic_path = snapshot_download(
            repo_id="ai4bharat/indictrans2-en-indic-dist-200M",
            revision=revision,
            cache_dir=cache_dir, 
            # local_dir_use_symlinks=False, # Not needed with cache_dir
            # local_files_only=True # Optional: ensure it doesn't try network if files exist (but they won't on first download)
        )
        print(f"EN→Indic model downloaded successfully to: {en_indic_path}")

        print("Downloading IndicTrans2 Indic→EN dist-200M model...")
        indic_en_path = snapshot_download(
            repo_id="ai4bharat/indictrans2-indic-en-dist-200M",
            revision=revision,
            cache_dir=cache_dir, 
            # local_dir_use_symlinks=False,
        )
        print(f"Indic→EN model downloaded successfully to: {indic_en_path}")

        print("All translation models downloaded to standard cache (dist-200M only)")
        print("TTS will use ElevenLabs API - no local TTS models needed")
        
        # Verify the downloads
        import glob
        model_files = glob.glob(f"{cache_dir}/models--ai4bharat--indictrans2-*/**", recursive=True)
        print(f"Verification: Found {len(model_files)} model files in cache")
        
    except Exception as e:
        print(f"ERROR: Failed to download models: {e}")
        raise

if __name__ == "__main__":
    main()