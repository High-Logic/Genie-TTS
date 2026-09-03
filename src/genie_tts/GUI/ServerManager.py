from PySide6.QtCore import Signal, QThread

from .. import Internal


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
                Internal.load_character(
                    character_name=self.req['character_name'],
                    onnx_model_dir=self.req['onnx_model_dir'],
                    language=self.req['language'],
                )
                self.finished.emit(True, "导入角色完成", None)

            elif self.mode == 'set_reference_audio':
                Internal.set_reference_audio(
                    character_name=self.req['character_name'],
                    audio_path=self.req['audio_path'],
                    audio_text=self.req['audio_text'],
                    language=self.req['language'],
                )
                self.finished.emit(True, "设置参考音频完成", None)

            elif self.mode == 'tts':
                audio_chunk = Internal.tts_to_numpy(
                    character_name=self.req['character_name'],
                    text=self.req['text'],
                )
                audio_chunk = audio_chunk.squeeze()
                return_data = {
                    "sample_rate": 32000,
                    "audio_list": [audio_chunk],
                }
                self.finished.emit(True, "推理完成", return_data)

        except Exception as e:
            self.finished.emit(False, f"请求异常: {str(e)}", None)
