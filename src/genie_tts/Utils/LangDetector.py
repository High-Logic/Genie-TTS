"""Language detection and mixed-language segmentation utilities.

Uses fast_langdetect for per-segment language detection.
Supported output languages: Japanese, English, Chinese, Korean.
Unknown or low-confidence detections fall back to English with a warning.
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)

# Mapping from fast_langdetect ISO 639-1 codes -> Genie canonical names
_FASTLANG_MAP: dict[str, str] = {
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
    # Additional codes fast_langdetect may return
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
}

_FALLBACK_LANG = "English"
_MIN_SEGMENT_LEN = 2  # minimum chars to keep a segment separate


class LangSegment(TypedDict):
    language: str
    content: str


def _load_detector():
    """Lazy-load fast_langdetect to avoid startup cost."""
    try:
        from fast_langdetect import detect
        return detect
    except ImportError:
        logger.warning(
            "fast_langdetect not installed. "
            "Install it with: pip install fast_langdetect. "
            "Falling back to English for auto-detection."
        )
        return None


_detect_fn = None


def detect_language(text: str) -> str:
    """Detect the dominant language of *text* and return a canonical language name.

    Returns one of: 'Japanese', 'English', 'Chinese', 'Korean'.
    Falls back to 'English' with a warning if detection fails or language is unsupported.
    """
    global _detect_fn
    if _detect_fn is None:
        _detect_fn = _load_detector()

    if not text or not text.strip():
        return _FALLBACK_LANG

    if _detect_fn is None:
        return _FALLBACK_LANG

    try:
        result = _detect_fn(text)
        # fast_langdetect returns a list of dicts: [{'lang': 'zh', 'score': 0.99}, ...]
        # Take the top result (highest score is first)
        if isinstance(result, list) and result:
            code = result[0].get("lang", "").lower()
        elif isinstance(result, dict):
            code = result.get("lang", "").lower()
        else:
            code = str(result).lower()

        canonical = _FASTLANG_MAP.get(code)
        if canonical is None:
            logger.warning(
                "Auto language detection: unsupported language code '%s' for text %r. "
                "Falling back to %s.",
                code, text[:40], _FALLBACK_LANG,
            )
            return _FALLBACK_LANG
        return canonical
    except Exception as exc:
        logger.warning(
            "Auto language detection failed for text %r: %s. Falling back to %s.",
            text[:40], exc, _FALLBACK_LANG,
        )
        return _FALLBACK_LANG


# Regex patterns for script-based pre-segmentation
_RE_CJK = re.compile(
    r"[\u3040-\u30ff"
    r"\u3400-\u4dbf"
    r"\u4e00-\u9fff"
    r"\uf900-\ufaff"
    r"\ufe30-\ufe4f"
    r"\uac00-\ud7a3"
    r"\u3130-\u318f]+"
)
_RE_LATIN = re.compile(r"[a-zA-Z][a-zA-Z\s'\-]*")


def segment_by_language(text: str) -> list[LangSegment]:
    """Split *text* into segments, each labelled with a detected language.

    Strategy:
    1. Pre-split on script boundaries (CJK vs Latin vs other).
    2. Detect language for each chunk using fast_langdetect.
    3. Merge adjacent same-language chunks.
    4. Drop empty segments; attach orphaned punctuation to preceding segment.

    Returns a list of LangSegment dicts: {language: str, content: str}.
    """
    if not text or not text.strip():
        return []

    # Split into alternating CJK / non-CJK runs, keeping delimiters
    parts = re.split(r"([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
                     r"\uf900-\ufaff\ufe30-\ufe4f\uac00-\ud7a3"
                     r"\u3130-\u318f]+)", text)

    raw: list[LangSegment] = []
    for part in parts:
        part = part  # keep original spacing
        if not part:
            continue
        if len(part.strip()) < _MIN_SEGMENT_LEN:
            # Too short to detect reliably — attach to previous if possible
            if raw:
                raw[-1]["content"] += part
            else:
                raw.append({"language": _FALLBACK_LANG, "content": part})
            continue
        lang = detect_language(part)
        raw.append({"language": lang, "content": part})

    if not raw:
        return [{"language": detect_language(text), "content": text}]

    # Merge adjacent same-language segments
    merged: list[LangSegment] = [raw[0]]
    for seg in raw[1:]:
        if seg["language"] == merged[-1]["language"]:
            merged[-1]["content"] += seg["content"]
        else:
            merged.append(seg)

    # Filter out empty segments
    return [s for s in merged if s["content"].strip()]
