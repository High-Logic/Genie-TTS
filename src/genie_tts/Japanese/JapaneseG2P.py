# -*- coding: utf-8 -*-
"""
用于纯日语的 G2P。
"""
# import os
# os.environ['OPEN_JTALK_DICT_DIR'] = './Data/open_jtalk_dic_utf_8-1.11'

import re
import pyopenjtalk
from typing import List
from .SymbolsV2 import symbols_v2, symbol_to_id_v2

# 匹配连续的标点符号
_CONSECUTIVE_PUNCTUATION_RE = re.compile(r"([,./?!~…・])\1+")

# 匹配需要转换为日语读法的特殊符号
_SYMBOLS_TO_JAPANESE = [
    (re.compile("%"), "パーセント"),
    (re.compile("％"), "パーセント"),
]

# 匹配日语字符（汉字、假名、全角字母数字等）
_JAPANESE_CHARACTERS_RE = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# 匹配非日语字符（标点、空格等）
_JAPANESE_MARKS_RE = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


class JapaneseG2P:
    """
    一个简化的、封装好的日语Grapheme-to-Phoneme（字素到音素）转换器。

    本版本假设 pyopenjtalk 库已安装，并且不使用任何用户自定义词典。
    它专注于提供一个纯粹、高效的文本到音素转换接口。
    """

    @staticmethod
    def _text_normalize(text: str) -> str:
        """对输入文本进行基础的规范化处理。"""
        for regex, replacement in _SYMBOLS_TO_JAPANESE:
            text = re.sub(regex, replacement, text)
        text = _CONSECUTIVE_PUNCTUATION_RE.sub(r"\1", text)
        text = text.lower()
        return text

    @staticmethod
    def _post_replace_phoneme(ph: str) -> str:
        """对单个音素或标点进行后处理替换。"""
        rep_map = {
            "：": ",", "；": ",", "，": ",", "。": ".",
            "！": "!", "？": "?", "\n": ".", "·": ",",
            "、": ",", "...": "…",
        }
        return rep_map.get(ph, ph)

    @staticmethod
    def _numeric_feature_by_regex(regex: str, s: str) -> int:
        """从OpenJTalk标签中提取数值特征。"""
        match = re.search(regex, s)
        return int(match.group(1)) if match else -50

    @staticmethod
    def _pyopenjtalk_g2p_prosody(text: str) -> List[str]:
        """使用pyopenjtalk提取音素及韵律符号。"""
        labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
        phones = []
        for n, lab_curr in enumerate(labels):
            p3 = re.search(r"-(.*?)\+", lab_curr).group(1)
            if p3 in "AEIOU":
                p3 = p3.lower()

            if p3 == "sil":
                if n == 0:
                    phones.append("^")
                elif n == len(labels) - 1:
                    e3 = JapaneseG2P._numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    phones.append("?" if e3 == 1 else "$")
                continue
            elif p3 == "pau":
                phones.append("_")
                continue
            else:
                phones.append(p3)

            a1 = JapaneseG2P._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = JapaneseG2P._numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = JapaneseG2P._numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
            f1 = JapaneseG2P._numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
            lab_next = labels[n + 1] if n + 1 < len(labels) else ""
            a2_next = JapaneseG2P._numeric_feature_by_regex(r"\+(\d+)\+", lab_next)

            if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                phones.append("#")
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                phones.append("]")
            elif a2 == 1 and a2_next == 2:
                phones.append("[")

        return phones

    @staticmethod
    def g2p(text: str, with_prosody: bool = True) -> List[str]:
        """
        将日语文本转换为音素序列。

        Args:
            text (str): 待转换的日语文本。
            with_prosody (bool): 是否在输出中包含韵律符号。默认为 True。

        Returns:
            List[str]: 音素和符号的列表。
        """
        if not text.strip():
            return []

        # 1. 文本规范化
        norm_text = JapaneseG2P._text_normalize(text)

        # 2. 使用标点符号分割字符串，得到日语文本片段
        japanese_segments = _JAPANESE_MARKS_RE.split(norm_text)
        punctuation_marks = _JAPANESE_MARKS_RE.findall(norm_text)

        phonemes = []
        for i, segment in enumerate(japanese_segments):
            if segment:
                if with_prosody:  # 移除分析结果中句首(^)/句尾($)的符号，因为我们按片段处理
                    phones = JapaneseG2P._pyopenjtalk_g2p_prosody(segment)[1:-1]
                else:
                    phones = pyopenjtalk.g2p(segment).split(" ")
                phonemes.extend(phones)

            # 将对应的标点符号添加回来
            if i < len(punctuation_marks):
                mark = punctuation_marks[i].strip()
                if mark:
                    phonemes.append(mark)

        # 3. 对最终列表中的每个元素进行后处理（主要转换全角标点）
        processed_phonemes = [JapaneseG2P._post_replace_phoneme(p) for p in phonemes]

        return processed_phonemes


def japanese_to_phones(text: str) -> list[int]:
    phones = JapaneseG2P.g2p(text)
    phones = ["UNK" if ph not in symbols_v2 else ph for ph in phones]
    phones = [symbol_to_id_v2[ph] for ph in phones]
    return phones
