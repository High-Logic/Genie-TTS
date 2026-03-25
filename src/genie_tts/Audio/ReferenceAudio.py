from ..Utils.Utils import LRUCacheDict
from ..GetPhonesAndBert import get_phones_and_bert
from ..Audio.Audio import load_audio
from ..ModelManager import model_manager

from onnxruntime import InferenceSession
import os
import numpy as np
import soxr
from typing import Optional, Dict


class ReferenceAudio:
    _prompt_cache: Dict[str, 'ReferenceAudio'] = LRUCacheDict(
        capacity=int(os.getenv('Max_Cached_Reference_Audio', '10')))

    def __new__(cls, prompt_wav: str, prompt_text: str, language: str):
        if prompt_wav in cls._prompt_cache:
            instance = cls._prompt_cache[prompt_wav]
            # Re-run G2P if text OR language changed since the entry was cached.
            if instance.text != prompt_text or getattr(instance, 'language', None) != language:
                instance.set_text(prompt_text, language=language)
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[prompt_wav] = instance
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str):
        if hasattr(self, '_initialized'):
            return

        # 文本相关。
        self.text: str = prompt_text
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        self.set_text(prompt_text, language=language)

        # 音频相关。
        self.audio_32k: Optional[np.ndarray] = load_audio(
            audio_path=prompt_wav,
            target_sampling_rate=32000
        )
        self.audio_16k: np.ndarray = soxr.resample(self.audio_32k, 32000, 16000, quality='hq')

        self.audio_32k = np.expand_dims(self.audio_32k, axis=0)
        self.audio_16k = np.expand_dims(self.audio_16k, axis=0)  # 增加 Batch_Size 维度

        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
        self.ssl_content: Optional[np.ndarray] = model_manager.cn_hubert.run(
            None, {'input_values': self.audio_16k}
        )[0]

        self.global_emb: Optional[np.ndarray] = None
        self.global_emb_advanced: Optional[np.ndarray] = None

        self._initialized = True

    def set_text(self, prompt_text: str, language: str) -> None:
        self.text = prompt_text
        self.language = language
        self.phonemes_seq, self.text_bert = get_phones_and_bert(prompt_text, language=language)

    @classmethod
    def clear_cache(cls) -> None:
        """清空 ReferenceAudio 的缓存"""
        cls._prompt_cache.clear()

    def update_global_emb(self, prompt_encoder: InferenceSession) -> None:
        if self.global_emb is not None:
            return
        if model_manager.load_sv_model():
            sv_emb = model_manager.speaker_verification_model.run(None, {'waveform': self.audio_16k})[0]
            self.global_emb, self.global_emb_advanced = prompt_encoder.run(None, {
                'ref_audio': self.audio_32k,
                'sv_emb': sv_emb,
            })
