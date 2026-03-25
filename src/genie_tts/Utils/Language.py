"""Language code normalisation utilities.

Canonical language names used throughout Genie-TTS:
  'Japanese', 'English', 'Chinese', 'Korean',
  'Hybrid-Chinese-English', 'Cantonese' (reserved), 'auto'
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Map of accepted aliases -> canonical name
_LANGUAGE_MAP: dict[str, str] = {
    # Chinese (Mandarin)
    "chinese": "Chinese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cmn": "Chinese",      # ISO 639-3 Mandarin
    "zho": "Chinese",      # ISO 639-3 Chinese (generic)
    "chi": "Chinese",      # ISO 639-2/B

    # English
    "english": "English",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "eng": "English",      # ISO 639-3

    # Japanese
    "japanese": "Japanese",
    "jp": "Japanese",
    "ja": "Japanese",
    "jpn": "Japanese",     # ISO 639-3
    "nihongo": "Japanese",

    # Korean
    "korean": "Korean",
    "ko": "Korean",
    "kr": "Korean",
    "kor": "Korean",       # ISO 639-3
    "hangul": "Korean",

    # Cantonese (symbols present in SymbolsV2; G2P pipeline not yet implemented)
    "cantonese": "Cantonese",
    "yue": "Cantonese",    # ISO 639-3 Yue Chinese
    "zh-yue": "Cantonese",

    # Hybrid modes
    "hybrid": "Hybrid-Chinese-English",
    "hybrid-zh-en": "Hybrid-Chinese-English",
    "hybrid-en-zh": "Hybrid-Chinese-English",
    "hybrid-chinese-english": "Hybrid-Chinese-English",

    # Auto-detection
    "auto": "auto",
}

# Languages with full G2P pipeline support
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({
    "Japanese", "English", "Chinese", "Korean",
    "Hybrid-Chinese-English", "auto",
})

# Languages that are recognised but not fully supported
_RESERVED_LANGUAGES: frozenset[str] = frozenset({"Cantonese"})


def normalize_language(lang: str) -> str:
    """Normalise *lang* to a canonical language name.

    Raises ValueError for completely unknown codes.
    Logs a warning for reserved (recognised but unsupported) languages.
    """
    canonical = _LANGUAGE_MAP.get(lang.lower().strip())
    if canonical is None:
        supported = sorted(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unknown language code: {lang!r}. "
            f"Supported languages: {supported}. "
            f"Use 'auto' for automatic detection."
        )
    if canonical in _RESERVED_LANGUAGES:
        logger.warning(
            "Language %r (%s) is recognised but not yet fully supported. "
            "G2P pipeline is not implemented. Consider using 'auto' or another supported language.",
            lang, canonical,
        )
    return canonical
