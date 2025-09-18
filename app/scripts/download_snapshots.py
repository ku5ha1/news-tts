import os
from huggingface_hub import snapshot_download


def main() -> None:
    revision = os.getenv("MODEL_REV") or None
    base_dir = "/app/models"
    os.makedirs(base_dir, exist_ok=True)
    print(f"Downloading dist-200M model snapshots into {base_dir}")

    print("Downloading IndicTrans2 EN→Indic dist-200M model...")
    snapshot_download(
        repo_id="ai4bharat/indictrans2-en-indic-dist-200M",
        revision=revision,
        local_dir=f"{base_dir}/indictrans2-en-indic-dist-200M",
        local_dir_use_symlinks=False,
    )
    
    print("Downloading IndicTrans2 Indic→EN dist-200M model...")
    snapshot_download(
        repo_id="ai4bharat/indictrans2-indic-en-dist-200M",
        revision=revision,
        local_dir=f"{base_dir}/indictrans2-indic-en-dist-200M",
        local_dir_use_symlinks=False,
    )
    
    print("All translation models ready (dist-200M only)")
    print("TTS will use ElevenLabs API - no local TTS models needed")


if __name__ == "__main__":
    main()
