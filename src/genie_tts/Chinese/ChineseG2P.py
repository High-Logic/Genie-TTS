# -*- coding: utf-8 -*-
"""
A universal G2P module that handles Chinese.
"""
import re
import pypinyin
from pypinyin.style._utils import get_initials, get_finals
from typing import List, Tuple
import numpy as np
from .SymbolsV2 import symbols_v2, symbol_to_id_v2

class ChineseG2P:
    """
    A G2P converter for Chinese.
    """
    @staticmethod
    def _chinese_g2p(text: str) -> List[str]:
        """Chinese G2P using pypinyin with correct decomposition."""
        phonemes = []
        pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
        
        for p_list in pinyins:
            syllable_with_tone = p_list[0]
            
            # Handle cases where pinyin is not found
            if syllable_with_tone == p_list[0] and not re.search(r'\d$', syllable_with_tone):
                phonemes.append(syllable_with_tone)
                continue

            syllable = syllable_with_tone.rstrip('12345')
            tone = syllable_with_tone[len(syllable):]

            if not tone:
                tone = '5'

            initial = get_initials(syllable, strict=False)
            final = get_finals(syllable, strict=False)

            if initial:
                phonemes.append(initial)
            if final:
                phonemes.append(f"{final}{tone}")
        
        return phonemes

    @staticmethod
    def g2p(text: str) -> List[str]:
        if not text.strip():
            return []
        return ChineseG2P._chinese_g2p(text)

def _g2p_single_syl(syl_with_tone: str) -> List[str]:
    """将单个拼音音节（带声调）转换为声母和韵母列表。"""
    phonemes = []
    syl = syl_with_tone.rstrip('12345')
    tone = syl_with_tone[len(syl):]
    if not tone:
        tone = '5'
    
    initial = get_initials(syl, strict=False)
    final = get_finals(syl, strict=False)
    
    if initial:
        phonemes.append(initial)
    if final:
        phonemes.append(f"{final}{tone}")
        
    return phonemes

def get_phonemes_and_bert(text: str, tokenizer, roberta_model) -> Tuple[np.ndarray, np.ndarray]:
    """
    为中文文本生成对齐的音素ID和BERT特征。
    """
    # 1. 使用RoBERTa模型获取BERT特征
    inputs = tokenizer(text, return_tensors="np")
    outputs = roberta_model.run(None, {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    })
    bert_output = np.squeeze(outputs[0], axis=0)  # Shape: (token_len, 768)
    
    # 2. 获取每个汉字的拼音
    pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    
    all_phonemes = []
    aligned_bert = []
    
    bert_idx = 1  # 从1开始，跳过[CLS]标记
    
    for i, p_list in enumerate(pinyins):
        syl_with_tone = p_list[0]
        
        # 如果是标点符号或非拼音字符
        if syl_with_tone == text[i]:
            phonemes_for_char = [syl_with_tone]
        else:
            phonemes_for_char = _g2p_single_syl(syl_with_tone)
        
        all_phonemes.extend(phonemes_for_char)
        
        # 将当前字符的BERT特征重复N次（N=该字符的音素数量）
        if bert_idx < bert_output.shape[0] - 1: # 确保不越界（跳过[SEP]）
            bert_feature = bert_output[bert_idx]
            aligned_bert.extend([bert_feature] * len(phonemes_for_char))
            bert_idx += 1
        else: # 处理末尾的特殊情况
            bert_feature = bert_output[-2] # 使用最后一个有效字符的特征
            aligned_bert.extend([bert_feature] * len(phonemes_for_char))

    # 3. 将对齐的BERT特征和音素转换为最终格式
    aligned_bert_np = np.array(aligned_bert, dtype=np.float32)
    
    # 填充维度至1024
    target_dim = 1024
    current_dim = aligned_bert_np.shape[1]
    if current_dim < target_dim:
        padding_size = target_dim - current_dim
        padding = np.zeros((aligned_bert_np.shape[0], padding_size), dtype=np.float32)
        final_bert = np.concatenate((aligned_bert_np, padding), axis=1)
    else:
        final_bert = aligned_bert_np[:, :target_dim]

    # 将音素转换为ID
    phoneme_ids = [symbol_to_id_v2.get(ph, symbol_to_id_v2["UNK"]) for ph in all_phonemes]
    
    return np.array([phoneme_ids], dtype=np.int64), final_bert


def chinese_to_phones(text: str) -> list[int]:
    phones = ChineseG2P.g2p(text)
    # Replace any phoneme not in the symbol list with UNK
    valid_phones = []
    for ph in phones:
        if ph in symbol_to_id_v2:
            valid_phones.append(ph)
        else:
            # This is a fallback for safety, ideally all phonemes should be in the list
            valid_phones.append("UNK")
            
    ids = [symbol_to_id_v2[ph] for ph in valid_phones]
    return ids
