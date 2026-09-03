"""冒烟测试公共环境：构造桩资源目录，使 import 不触发下载交互与资源校验失败。

conftest 在测试模块导入前执行，故此处直接修改环境变量与内置 input，
保证任何测试模块首次 import genie_tts 时已处于受控状态。
"""

import builtins
import os
import tempfile

_DATA_DIR = os.path.join(tempfile.gettempdir(), "genie_tts_smoke_data")

# Resources.py 导入期仅校验 HUBERT_MODEL_DIR 与 SV_MODEL 存在
os.makedirs(os.path.join(_DATA_DIR, "chinese-hubert-base"), exist_ok=True)
with open(os.path.join(_DATA_DIR, "speaker_encoder.onnx"), "a"):
    pass

os.environ["GENIE_DATA_DIR"] = _DATA_DIR
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _no_input(*_args, **_kwargs):
    raise AssertionError("冒烟测试不应触发交互式下载")


builtins.input = _no_input
