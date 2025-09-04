from langdetect import detect

def detect_language(text: str) -> str:
    lang = detect(text)
    if lang.startswith("hi"):
        return "hi"
    elif lang.startswith("kn"):
        return "kn"
    else:
        return "en"