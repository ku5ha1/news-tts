from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TranslationService:
    def __init__(self):
        # Load both models with trust_remote_code=True
        self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
            "ai4bharat/IndicTrans2-en-indic-1B", 
            trust_remote_code=True
        )
        self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
            "ai4bharat/IndicTrans2-en-indic-1B", 
            trust_remote_code=True
        )

        self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
            "ai4bharat/IndicTrans2-indic-en-1B", 
            trust_remote_code=True
        )
        self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
            "ai4bharat/IndicTrans2-indic-en-1B", 
            trust_remote_code=True
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_en_indic.to(self.device)
        self.model_indic_en.to(self.device)

    def _translate(self, text: str, source: str, target: str) -> str:
        # target tokens: hi / kn / en
        target_token = f"<2{target}> {text}"

        if source == "en":
            tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
        else:
            tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

        inputs = tokenizer(target_token, return_tensors="pt", padding=True).to(self.device)
        outputs = model.generate(**inputs, max_length=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_to_all(self, title: str, description: str, source_lang: str):
        """Translate given text to the other two languages"""
        languages = ["en", "hi", "kn"]
        target_languages = [l for l in languages if l != source_lang]

        result = {}
        for lang in target_languages:
            title_trans = self._translate(title, source_lang, lang)
            desc_trans = self._translate(description, source_lang, lang)
            result[lang] = {"title": title_trans, "description": desc_trans}
        return result
