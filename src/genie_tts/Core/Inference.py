import onnxruntime as ort
import numpy as np
from typing import List, Optional
import threading

from ..Audio.ReferenceAudio import ReferenceAudio
from ..GetPhonesAndBert import get_phones_and_bert

MAX_T2S_LEN = 1000


# 中文说明：
# ONNX 有时会把 stop condition 返回成 0-d/1-d ndarray，而不是纯 Python bool。
# 如果直接写 `if stop_condition_tensor:`，在不同 provider / 不同返回形态下，
# 可能出现判断不稳定，甚至触发数组真值判断歧义。
def _should_stop_decoding(stop_condition_tensor: object) -> bool:
    stop_array = np.asarray(stop_condition_tensor)
    if stop_array.size == 0:
        return False
    return bool(stop_array.reshape(-1)[0])


# 中文说明：
# 这里只允许把“本轮真正新生成”的 semantic token 送进 vocoder。
# 之前的实现使用 `y[:, -idx:]` 截取尾部 token，看起来像是在取最后 idx 个，
# 但当 idx == 0 时，NumPy 的 `-0` 仍然是 0，`y[:, -0:]` 会退化成“整段全取”。
# 一旦 decoder 在第一轮就 stop，就会把包含 prompt/ref 部分的整段语义错误送去合成，
# 最终表现成把 ref_text / prompt 内容说出来。
#
# 修复策略：
# - 单独统计本轮真正生成了多少步 generated_steps
# - 只截最后 generated_steps 个 token
# - 如果 generated_steps <= 0，直接明确报错，不再悄悄回退成整段 prompt
#   这样可以把“没有生成内容”和“生成了错误内容”这两类问题区分开

def _extract_generated_semantic_tokens(y: np.ndarray, generated_steps: int) -> np.ndarray:
    if generated_steps <= 0:
        raise RuntimeError('GENIE decoder returned zero generated semantic steps')
    if y.ndim != 2:
        raise ValueError(f'Expected semantic tokens with 2 dims, got shape={y.shape}')

    slice_width = min(generated_steps, y.shape[1])
    return np.expand_dims(y[:, -slice_width:], axis=0)


class GENIE:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            prompt_encoder: Optional[ort.InferenceSession],
            language: str = 'japanese',
    ) -> Optional[np.ndarray]:
        text = '。' + text  # 防止漏第一句。
        text_seq, text_bert = get_phones_and_bert(text, language=language)

        semantic_tokens: np.ndarray = self.t2s_cpu(
            ref_seq=prompt_audio.phonemes_seq,
            ref_bert=prompt_audio.text_bert,
            text_seq=text_seq,
            text_bert=text_bert,
            ssl_content=prompt_audio.ssl_content,
            encoder=encoder,
            first_stage_decoder=first_stage_decoder,
            stage_decoder=stage_decoder,
        )
        # 中文说明：
        # t2s_cpu() 在收到 stop_event 时会返回 None。
        # 这里必须立刻向上返回，避免后面继续对 None 执行 np.where(...)，
        # 把“主动中断”变成另一类无关异常。
        if semantic_tokens is None:
            return None

        eos_indices = np.where(semantic_tokens >= 1024)  # 剔除不合法的元素，例如 EOS Token。
        if len(eos_indices[0]) > 0:
            first_eos_index = eos_indices[-1][0]
            semantic_tokens = semantic_tokens[..., :first_eos_index]

        if prompt_encoder is None:
            return vocoder.run(None, {
                "text_seq": text_seq,
                "pred_semantic": semantic_tokens,
                "ref_audio": prompt_audio.audio_32k
            })[0]
        else:
            # V2ProPlus 新增。
            prompt_audio.update_global_emb(prompt_encoder=prompt_encoder)
            audio_chunk = vocoder.run(None, {
                "text_seq": text_seq,
                "pred_semantic": semantic_tokens,
                "ge": prompt_audio.global_emb,
                "ge_advanced": prompt_audio.global_emb_advanced,
            })[0]
            return audio_chunk

    def t2s_cpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
    ) -> Optional[np.ndarray]:
        """在CPU上运行T2S模型"""
        # Encoder
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )

        # First Stage Decoder
        y, y_emb, *present_key_values = first_stage_decoder.run(
            None, {"x": x, "prompts": prompts}
        )

        # Stage Decoder
        input_names: List[str] = [inp.name for inp in stage_decoder.get_inputs()]
        # 中文说明：
        # generated_steps 记录“真正生成了多少轮新 token”，
        # 它和循环变量不是一回事：
        # - 循环变量会受到首轮 stop / break 时机影响
        # - 如果直接拿旧 idx 去切片，会在 idx == 0 时踩到 `y[:, -0:]` 全取陷阱
        generated_steps = 0
        for _ in range(0, 500):
            if self.stop_event.is_set():
                return None
            input_feed = {
                name: data
                for name, data in zip(input_names, [y, y_emb, *present_key_values])
            }
            outputs = stage_decoder.run(None, input_feed)
            y, y_emb, stop_condition_tensor, *present_key_values = outputs
            generated_steps += 1

            # 中文说明：
            # 这里统一通过 helper 读取 stop 条件，避免 ONNX 返回 ndarray 时的布尔判断不稳定。
            if _should_stop_decoding(stop_condition_tensor):
                break

        # 中文说明：
        # 保持原有行为：把最后一个位置清成 0，避免 EOS 等特殊 token 残留到后续路径。
        y[0, -1] = 0
        # 中文说明：
        # 只返回“本轮新生成”的 semantic token。
        # 如果实际上一轮都没生成，会在 helper 里明确报错，避免错误地把 prompt 当结果返回。
        return _extract_generated_semantic_tokens(y, generated_steps)


tts_client: GENIE = GENIE()
