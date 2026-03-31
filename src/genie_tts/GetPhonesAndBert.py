import numpy as np
import re
from typing import Tuple, Literal
from .Utils.Constants import BERT_FEATURE_DIM
from .ModelManager import model_manager


def _build_roberta_inputs(encoded, word2ph: list[int]) -> dict[str, np.ndarray]:
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    input_names = {inp.name for inp in model_manager.roberta_model.get_inputs()}

    ort_inputs: dict[str, np.ndarray] = {}
    if "input_ids" in input_names:
        ort_inputs["input_ids"] = input_ids
    if "attention_mask" in input_names:
        ort_inputs["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
    if "repeats" in input_names:
        ort_inputs["repeats"] = np.array(word2ph, dtype=np.int64)
    return ort_inputs


def _expand_roberta_output(text_bert: np.ndarray, word2ph: list[int], num_phones: int) -> np.ndarray:
    if text_bert.ndim == 3 and text_bert.shape[0] == 1:
        text_bert = text_bert[0]

    if text_bert.shape[0] == num_phones:
        return text_bert.astype(np.float32)

    if text_bert.shape[0] == len(word2ph) + 2:
        text_bert = text_bert[1:-1]

    if text_bert.shape[0] == len(word2ph):
        repeated = [
            np.repeat(text_bert[idx: idx + 1], repeats=count, axis=0)
            for idx, count in enumerate(word2ph)
        ]
        expanded = np.concatenate(repeated, axis=0)
        if expanded.shape[0] != num_phones:
            raise ValueError(
                f"Expanded RoBERTa features to {expanded.shape[0]} phones, expected {num_phones}"
            )
        return expanded.astype(np.float32)

    raise ValueError(
        "Unsupported RoBERTa output layout: "
        f"got first dimension {text_bert.shape[0]}, expected {num_phones} phones "
        f"or {len(word2ph)} / {len(word2ph) + 2} token features"
    )


def split_language(text: str) -> list[dict[Literal['language', 'content'], str]]:
    """
    从文本中提取中文和英文部分，返回一个包含语言和内容的列表。

    ### 参数:
    text (str): 输入的文本，包含中文和英文混合。

    ### 返回:
    list[dict[Literal['language', 'content'], str]]: 一个列表，每个元素是一个字典，包含语言（'chinese'或'english'）和对应的内容。
    """

    pattern_eng = re.compile(r"[a-zA-Z]+")
    split = re.split(pattern_eng, text)
    matches = pattern_eng.findall(text)

    assert len(matches) == len(split) - 1, "Mismatch between number of English matches and Chinese parts"

    result = []
    for i, part in enumerate(split):
        if part.strip():
            result.append({'language': 'chinese', 'content': part})
        if i < len(matches):
            result.append({'language': 'english', 'content': matches[i]})

    return result

def get_phones_and_bert(prompt_text: str, language: str = 'japanese') -> Tuple[np.ndarray, np.ndarray]:
    """获取 phones 序列和 bert 特征, 考虑混合语言问题"""

    if language.lower() == 'hybrid-chinese-english':
        split = split_language(prompt_text)

        list_phones = []
        list_berts = []

        for chunk in split:
            phones_seq, text_bert = _get_phones_and_bert_pure_lang(chunk['content'], chunk['language'])
            list_phones.append(phones_seq)
            list_berts.append(text_bert)

        phones_seq = np.concatenate(list_phones, axis=1)
        text_bert = np.concatenate(list_berts, axis=0)
    else:
        phones_seq, text_bert = _get_phones_and_bert_pure_lang(prompt_text, language)

    return phones_seq, text_bert

def _get_phones_and_bert_pure_lang(prompt_text: str, language: str = 'japanese') -> Tuple[np.ndarray, np.ndarray]:
    """获取 phones 序列和 bert 特征，不考虑混合语言问题"""

    if language.lower() == 'english':
        from .G2P.English.EnglishG2P import english_to_phones
        phones = english_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)
    elif language.lower() == 'chinese':
        from .G2P.Chinese.ChineseG2P import chinese_to_phones
        text_clean, _, phones, word2ph = chinese_to_phones(prompt_text)
        if model_manager.load_roberta_model():
            encoded = model_manager.roberta_tokenizer.encode(text_clean)
            ort_inputs = _build_roberta_inputs(encoded, word2ph)
            outputs = model_manager.roberta_model.run(None, ort_inputs)
            text_bert = _expand_roberta_output(outputs[0], word2ph, len(phones))
        else:
            text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)
    elif language.lower() == 'korean':
        from .G2P.Korean.KoreanG2P import korean_to_phones
        phones = korean_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)
    else:
        from .G2P.Japanese.JapaneseG2P import japanese_to_phones
        phones = japanese_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    phones_seq = np.array([phones], dtype=np.int64)
    return phones_seq, text_bert
