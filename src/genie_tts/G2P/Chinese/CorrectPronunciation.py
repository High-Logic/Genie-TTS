import os
import pickle
from typing import List, Dict, Any, Union

from ...Core.Resources import Chinese_G2P_DIR

# 常量定义
DEFAULT_CACHE_PATH = os.path.join(Chinese_G2P_DIR, "polyphonic.pickle")


class PolyphonicDictManager:
    _data: Dict[str, Any] = {}

    @classmethod
    def get_data(cls, path: str = DEFAULT_CACHE_PATH) -> Dict[str, Any]:
        if not cls._data:
            with open(path, "rb") as f:
                cls._data = pickle.load(f)
        return cls._data


def correct_pronunciation(word: str, word_pinyin: List[str]) -> Union[List[str], str]:
    """
        根据加载的字典修正发音，作为供外部程序调用的独立接口。
        逻辑：优先查找整词修正，如果没有整词匹配，则遍历每个字符进行单字修正。

        Input:
            word (str): 原始中文字符串，例如 "银行"。
            word_pinyins (List[str]): 当前预测的拼音列表，例如 ['yin2', 'xing2']。

        Output:
            Union[List[str], str]: 修正后的拼音列表或字符串。

        Example:
            # 字典包含整词 {'银行': ['yin2', 'hang2']}
            result = correct_pronunciation("银行", ["yin2", "xing2"])
            # Result: ["yin2", "hang2"]
        """
    pp_dict = PolyphonicDictManager.get_data()
    new_word_pinyin = list(word_pinyin)
    # 1. 尝试整词匹配
    if new_pinyin := pp_dict.get(word):
        return new_pinyin
    # 2. 逐字修正
    for idx, w in enumerate(word):
        if idx >= len(new_word_pinyin):
            break
        if w_pinyin := pp_dict.get(w):
            new_word_pinyin[idx] = w_pinyin[0]
    return new_word_pinyin
