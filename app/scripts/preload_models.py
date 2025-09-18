import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from parler_tts import ParlerTTSForConditionalGeneration

def main() -> None:
    model_size = os.getenv("MODEL_SIZE", "dist-200M")
    base_en_indic = f"ai4bharat/indictrans2-en-indic-{model_size}"
    base_indic_en = f"ai4bharat/indictrans2-indic-en-{model_size}"
    tts_model = "ai4bharat/indic-parler-tts"
    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")

    rev = os.getenv("MODEL_REV")

    print(f"Preloading translation models: {base_en_indic} and {base_indic_en}")

    AutoTokenizer.from_pretrained(base_en_indic, trust_remote_code=True, cache_dir=cache_dir, revision=rev)
    AutoModelForSeq2SeqLM.from_pretrained(base_en_indic, trust_remote_code=True, cache_dir=cache_dir, revision=rev)

    AutoTokenizer.from_pretrained(base_indic_en, trust_remote_code=True, cache_dir=cache_dir, revision=rev)
    AutoModelForSeq2SeqLM.from_pretrained(base_indic_en, trust_remote_code=True, cache_dir=cache_dir, revision=rev)

    print(f"Preloading TTS model: {tts_model}")
    AutoTokenizer.from_pretrained(tts_model, trust_remote_code=True, cache_dir=cache_dir, revision=rev)
    ParlerTTSForConditionalGeneration.from_pretrained(tts_model, trust_remote_code=True, cache_dir=cache_dir, revision=rev)

    print("Done preloading all models (Translation + TTS).")

if __name__ == "__main__":
    main()
