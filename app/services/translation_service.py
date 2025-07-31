from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from pathlib import Path

class TranslationService:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self._load_models()
            TranslationService._models_loaded = True
    
    def _load_models(self):
        """Load models with caching"""
        try:
            # Set cache directory for models
            cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            print("Loading AI4Bharat translation models...")
            
            # Load both models with trust_remote_code=True and caching
            self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicTrans2-en-indic-1B", 
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False  # Allow downloading if not cached
            )
            self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/IndicTrans2-en-indic-1B", 
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False
            )

            self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicTrans2-indic-en-1B", 
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/IndicTrans2-indic-en-1B", 
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # Move models to device
            self.model_en_indic.to(self.device)
            self.model_indic_en.to(self.device)
            
            # Set models to evaluation mode
            self.model_en_indic.eval()
            self.model_indic_en.eval()
            
            print("AI4Bharat translation models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading translation models: {str(e)}")
            raise e
    
    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate text between languages"""
        try:
            # target tokens: hi / kn / en
            target_token = f"<2{target}> {text}"

            if source == "en":
                tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
            else:
                tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

            with torch.no_grad():  # Disable gradient computation for inference
                inputs = tokenizer(target_token, return_tensors="pt", padding=True).to(self.device)
                outputs = model.generate(**inputs, max_length=256)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails

    def translate_to_all(self, title: str, description: str, source_lang: str):
        """Translate given text to the other two languages"""
        try:
            languages = ["en", "hi", "kn"]
            target_languages = [l for l in languages if l != source_lang]

            result = {}
            for lang in target_languages:
                title_trans = self._translate(title, source_lang, lang)
                desc_trans = self._translate(description, source_lang, lang)
                result[lang] = {"title": title_trans, "description": desc_trans}
            return result
            
        except Exception as e:
            print(f"Error in translate_to_all: {str(e)}")
            return {}
    
    def is_models_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded and hasattr(self, 'model_en_indic')
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
            "models_loaded": self.is_models_loaded(),
            "device": str(self.device) if hasattr(self, 'device') else "unknown",
            "en_indic_model": "ai4bharat/IndicTrans2-en-indic-1B",
            "indic_en_model": "ai4bharat/IndicTrans2-indic-en-1B"
        }
