import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main() -> None:
    
    base_en_indic = "ai4bharat/indictrans2-en-indic-dist-200M"
    base_indic_en = "ai4bharat/indictrans2-indic-en-dist-200M"
    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
    
    rev = os.getenv("MODEL_REV")

    print(f"Preloading translation models: {base_en_indic} and {base_indic_en}")

    print("Loading EN→Indic tokenizer...")
    AutoTokenizer.from_pretrained(base_en_indic, trust_remote_code=True, cache_dir=cache_dir, revision=rev)
    
    print("Loading EN→Indic model...")
    AutoModelForSeq2SeqLM.from_pretrained(base_en_indic, trust_remote_code=True, cache_dir=cache_dir, revision=rev)

    print("Loading Indic→EN tokenizer...")
    AutoTokenizer.from_pretrained(base_indic_en, trust_remote_code=True, cache_dir=cache_dir, revision=rev)
    
    print("Loading Indic→EN model...")
    AutoModelForSeq2SeqLM.from_pretrained(base_indic_en, trust_remote_code=True, cache_dir=cache_dir, revision=rev)

    print("Done preloading translation models (dist-200M only)")


if __name__ == "__main__":
    main()
