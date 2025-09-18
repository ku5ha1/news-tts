import os
from huggingface_hub import snapshot_download

def main() -> None:
    revision = os.getenv("MODEL_REV") or None
    base_dir = "/app/models"
    os.makedirs(base_dir, exist_ok=True)
    print(f"Downloading model snapshots into {base_dir}")

    print("Downloading IndicTrans2 EN→Indic model...")
    snapshot_download(
        repo_id="ai4bharat/indictrans2-en-indic-dist-200M",
        revision=revision,
        local_dir=f"{base_dir}/indictrans2-en-indic-dist-200M",
        local_dir_use_symlinks=False,
    )
    
    print("Downloading IndicTrans2 Indic→EN model...")
    snapshot_download(
        repo_id="ai4bharat/indictrans2-indic-en-dist-200M",
        revision=revision,
        local_dir=f"{base_dir}/indictrans2-indic-en-dist-200M",
        local_dir_use_symlinks=False,
    )
    
    # add parler tts to baked image
    print("Downloading Indic Parler-TTS model...")
    snapshot_download(
        repo_id="ai4bharat/indic-parler-tts",
        revision=revision,
        local_dir=f"{base_dir}/indic-parler-tts",
        local_dir_use_symlinks=False,
    )
    
    print("All snapshots ready (Translation + TTS models)")

if __name__ == "__main__":
    main()