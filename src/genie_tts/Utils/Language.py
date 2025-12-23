language_map = {
    # Chinese
    "chinese": "Chinese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",

    # English
    "english": "English",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "eng": "English",

    # Japanese
    "japanese": "Japanese",
    "jp": "Japanese",
    "ja": "Japanese",
    "nihongo": "Japanese",

    # Hybrid
    "hybrid": "Hybrid-Chinese-English",
    "hybrid-zh-en": "Hybrid-Chinese-English",
    "hybrid-en-zh": "Hybrid-Chinese-English",

    # Korean
    "korean": "Korean",
    "ko": "Korean",
    "kr": "Korean",
    "hangul": "Korean",
    
}


def normalize_language(lang: str) -> str:
    return language_map.get(lang.lower(), lang)
