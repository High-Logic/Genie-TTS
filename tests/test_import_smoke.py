"""契约冒烟：公共 API 与入口模块可导入。

仅断言契约面（__all__ 集合与入口模块路径），不涉及内部模块路径；
无模型数据、无 torch 的环境下须全部通过。
"""

import importlib

import pytest

PUBLIC_API = {
    "load_character",
    "unload_character",
    "set_reference_audio",
    "tts_async",
    "tts",
    "stop",
    "convert_to_onnx",
    "clear_reference_audio_cache",
    "start_server",
    "load_predefined_character",
    "wait_for_playback_done",
    "download_genie_data",
    "download_roberta_data",
}


def test_public_api():
    genie_tts = importlib.import_module("genie_tts")
    public_api = set(genie_tts.__all__)
    assert PUBLIC_API.issubset(public_api), f"公共 API 缺失: {PUBLIC_API - public_api}"
    for name in PUBLIC_API:
        assert hasattr(genie_tts, name), f"公共 API 缺失: {name}"


def test_server_entry():
    importlib.import_module("genie_tts.Server")


def test_gui_offscreen():
    pytest.importorskip("PySide6")
    # 已知限制：GUI 顶层导入链（ConverterWidget → Converter）会急切 import torch，
    # 无 torch 环境无法导入 GUI；改为函数级导入后移除该检查
    pytest.importorskip("torch")
    importlib.import_module("genie_tts.GUI.GUI")


def test_converter_entry():
    pytest.importorskip("torch")
    importlib.import_module("genie_tts.Converter.Converter")
