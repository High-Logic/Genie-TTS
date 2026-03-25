"""Phone sequence and BERT feature extraction with multi-language dispatch.

Supported language values (after normalize_language()):
  'Japanese', 'English', 'Chinese', 'Korean',
  'Hybrid-Chinese-English', 'auto'
"""
from __future__ import annotations

import re
import logging
import numpy as np
from typing import Tuple

from .Utils.Constants import BERT_FEATURE_DIM
from .ModelManager import model_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy Hybrid-Chinese-English splitter (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def _split_chinese_english(text: str) -> list[dict]:
    """Split text into Chinese and English chunks via Latin-character regex.

    Kept for backward compatibility with 'Hybrid-Chinese-English' mode.
    For general mixed-language splitting, use Utils.LangDetector.segment_by_language.
    """
    pattern_eng = re.compile(r"[a-zA-Z]+")
    parts = re.split(pattern_eng, text)
    matches = pattern_eng.findall(text)

    result: list[dict] = []
    for i, part in enumerate(parts):
        if part.strip():
            result.append({"language": "chinese", "content": part})
        if i < len(matches):
            result.append({"language": "english", "content": matches[i]})
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_phones_and_bert(
    prompt_text: str, language: str = "japanese"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (phones_seq, text_bert) for *prompt_text* in *language*.

    *language* should already be normalised by normalize_language().
    Handles multi-language modes ('Hybrid-Chinese-English', 'auto') by
    splitting text into per-language chunks and concatenating results.
    """
    lang_lower = language.lower()

    if lang_lower == "hybrid-chinese-english":
        chunks = _split_chinese_english(prompt_text)
        return _process_chunks(chunks)

    if lang_lower == "auto":
        from .Utils.LangDetector import segment_by_language
        chunks = segment_by_language(prompt_text)
        if not chunks:
            logger.warning("LangDetector returned no segments for text %r; falling back to Japanese.", prompt_text[:60])
            return _get_phones_and_bert_single(prompt_text, "japanese")
        if len(chunks) == 1:
            return _get_phones_and_bert_single(chunks[0]["content"], chunks[0]["language"])
        return _process_chunks(chunks)

    return _get_phones_and_bert_single(prompt_text, language)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_chunks(chunks: list[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Run G2P on each chunk and concatenate results."""
    list_phones: list[np.ndarray] = []
    list_berts: list[np.ndarray] = []
    for chunk in chunks:
        phones_seq, text_bert = _get_phones_and_bert_single(
            chunk["content"], chunk["language"]
        )
        list_phones.append(phones_seq)
        list_berts.append(text_bert)
    phones_seq = np.concatenate(list_phones, axis=1)
    text_bert = np.concatenate(list_berts, axis=0)
    return phones_seq, text_bert


def _get_phones_and_bert_single(
    prompt_text: str, language: str = "japanese"
) -> Tuple[np.ndarray, np.ndarray]:
    """Run G2P for a single-language text chunk."""
    lang_lower = language.lower()

    if lang_lower == "english":
        from .G2P.English.EnglishG2P import english_to_phones
        phones = english_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    elif lang_lower == "chinese":
        from .G2P.Chinese.ChineseG2P import chinese_to_phones
        text_clean, _, phones, word2ph = chinese_to_phones(prompt_text)
        if model_manager.load_roberta_model():
            encoded = model_manager.roberta_tokenizer.encode(text_clean)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "repeats": np.array(word2ph, dtype=np.int64),
            }
            outputs = model_manager.roberta_model.run(None, ort_inputs)
            text_bert = outputs[0].astype(np.float32)
        else:
            text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    elif lang_lower == "korean":
        from .G2P.Korean.KoreanG2P import korean_to_phones
        phones = korean_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    else:
        if lang_lower not in ("japanese",):
            logger.warning(
                "Unsupported language %r in _get_phones_and_bert_single; "
                "falling back to Japanese G2P.",
                language,
            )
        from .G2P.Japanese.JapaneseG2P import japanese_to_phones
        phones = japanese_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    phones_seq = np.array([phones], dtype=np.int64)
    return phones_seq, text_bert
