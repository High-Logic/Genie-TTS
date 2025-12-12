import numpy as np
from PySide6.QtCore import Signal, QThread

from ..Utils.Shared import context
from ..Internal import load_character, set_reference_audio
from ..Core.Inference import tts_client
from ..ModelManager import model_manager


class InferenceWorker(QThread):
    """执行推理任务的 Worker"""
    finished = Signal(bool, str, object)  # success, message, data

    def __init__(self, request_data: dict, mode: str):
        super().__init__()
        self.req: dict = request_data
        self.mode: str = mode

    def run(self) -> None:
        try:
            if self.mode == 'load_character':
                load_character(
                    character_name=self.req['character_name'],
                    onnx_model_dir=self.req['onnx_model_dir'],
                    language=self.req['language'],
                )

            elif self.mode == 'set_reference_audio':
                set_reference_audio(
                    character_name=self.req['character_name'],
                    audio_path=self.req['audio_path'],
                    audio_text=self.req['audio_text'],
                    language=self.req['language'],
                )

            elif self.mode == 'tts':
                gsv_model = model_manager.get(self.req['character_name'])
                tts_client.stop_event.clear()
                audio_chunk = tts_client.tts(
                    text=self.req['text'],
                    prompt_audio=context.current_prompt_audio,
                    encoder=gsv_model.T2S_ENCODER,
                    first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                    stage_decoder=gsv_model.T2S_STAGE_DECODER,
                    vocoder=gsv_model.VITS,
                    prompt_encoder=gsv_model.PROMPT_ENCODER,
                    language=gsv_model.LANGUAGE,
                )
                try:
                    return_data = {
                        "sample_rate": 32000,
                        "audio_list": [audio_chunk.astype(np.int16)],
                    }
                    self.finished.emit(True, "推理完成", return_data)
                except Exception as e:
                    self.finished.emit(False, f"数据解析失败: {e}", None)

        except Exception as e:
            self.finished.emit(False, f"请求异常: {str(e)}", None)
