from langdetect import detect

def detect_language(text: str) -> str:
    # Returns 'en', 'hi', or 'kn' (fallback to 'en' if unsupported)
    lang = detect(text)
    if lang.startswith("hi"):
        return "hi"
    elif lang.startswith("kn"):
        return "kn"
    else:
        return "en"
