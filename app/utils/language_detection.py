from langdetect import detect

LANGUAGE_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda"
}

def detect_language(text: str) -> str:
    lang = detect(text)
    if lang.startswith("hi"):
        return LANGUAGE_TAGS["hi"]
    elif lang.startswith("kn"):
        return LANGUAGE_TAGS["kn"]
    else:
        return LANGUAGE_TAGS["en"]
