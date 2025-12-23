# 请严格遵循导入顺序。
# 1、环境变量。
import os
from os import PathLike

os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"

# 2、Logging & Warnings。
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="jieba_fast._compat")
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)

# 3、ONNX。
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# 导入剩余库。

from pathlib import Path
import json
import asyncio
from typing import AsyncIterator, Optional, Union, Dict

from .Audio.ReferenceAudio import ReferenceAudio
from .Core.Resources import ensure_exists, Chinese_G2P_DIR, English_G2P_DIR
from .Core.TTSPlayer import tts_player
from .ModelManager import model_manager
from .Utils.Shared import context
from .Utils.Language import normalize_language
from .PredefinedCharacter import download_chara, CHARA_LANG, CHARA_ALIAS_MAP

# A module-level private dictionary to store reference audio configurations.
_reference_audios: Dict[str, dict] = {}
SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def check_onnx_model_dir(onnx_model_dir: Union[str, os.PathLike]) -> None:
    """
    Checks if the directory contains the necessary ONNX model files for Genie TTS (v2 or v2ProPlus).
    Raises a FileNotFoundError with detailed instructions if validation fails.
    """
    model_path = Path(onnx_model_dir)

    # 1. Check if directory exists
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(f"The model directory '{onnx_model_dir}' does not exist or is not a directory.")

    # 2. Define required files
    # Base files required by both v2 and v2ProPlus
    required_base_files = {
        "t2s_encoder_fp32.bin",
        "t2s_encoder_fp32.onnx",
        "t2s_first_stage_decoder_fp32.onnx",
        "t2s_shared_fp16.bin",
        "t2s_stage_decoder_fp32.onnx",
        "vits_fp16.bin",
        "vits_fp32.onnx"
    }

    # 3. Get current files in directory
    existing_files = set(f.name for f in model_path.iterdir() if f.is_file())

    # 4. Validate
    # We check if the base files exist. If base files are missing, the model is definitely unusable.
    if not required_base_files.issubset(existing_files):
        missing = required_base_files - existing_files

        # Construct detailed error message
        error_msg = (
            f"\n\n[Genie Error] Invalid ONNX model directory: '{model_path}'\n"
            "===============================================================\n"
            f"Missing base files: {', '.join(missing)}\n"
            "A valid model folder must contain at least the following files.\n"
            "1. [v2 Base] (Required for all models):\n"
            "   - t2s_encoder_fp32.bin\n"
            "   - t2s_encoder_fp32.onnx\n"
            "   - t2s_first_stage_decoder_fp32.onnx\n"
            "   - t2s_shared_fp16.bin\n"
            "   - t2s_stage_decoder_fp32.onnx\n"
            "   - vits_fp16.bin\n"
            "   - vits_fp32.onnx\n"
            "2. [v2ProPlus Additions] (Required for v2pp features):\n"
            "   - prompt_encoder_fp16.bin\n"
            "   - prompt_encoder_fp32.onnx\n"
            "===============================================================\n"
        )
        raise FileNotFoundError(error_msg)


def load_character(
        character_name: str,
        onnx_model_dir: Union[str, PathLike],
        language: str,
) -> None:
    """
    Loads a character model from an ONNX model directory.

    Args:
        character_name (str): The name to assign to the loaded character.
        onnx_model_dir (str | PathLike): The directory path containing the ONNX model files.
        language (str): The language of the character model.
    """
    check_onnx_model_dir(onnx_model_dir)

    language = normalize_language(language)
    if language not in ['Japanese', 'English', 'Chinese', 'Hybrid-Chinese-English', 'Korean']:
        raise ValueError('Unknown language')

    if language == 'Chinese':
        ensure_exists(Chinese_G2P_DIR, "Chinese_G2P_DIR")
    elif language == 'English':
        ensure_exists(English_G2P_DIR, "English_G2P_DIR")
    elif language == 'Hybrid-Chinese-English':
        ensure_exists(Chinese_G2P_DIR, "Chinese_G2P_DIR")
        ensure_exists(English_G2P_DIR, "English_G2P_DIR")

    model_path: str = os.fspath(onnx_model_dir)
    model_manager.load_character(
        character_name=character_name,
        model_dir=model_path,
        language=language,
    )


def unload_character(
        character_name: str,
) -> None:
    """
    Unloads a previously loaded character model to free up resources.

    Args:
        character_name (str): The name of the character to unload.
    """
    model_manager.remove_character(
        character_name=character_name,
    )


def set_reference_audio(
        character_name: str,
        audio_path: Union[str, PathLike],
        audio_text: str,
        language: str = None,
) -> None:
    """
    Sets the reference audio for a character to be used for voice cloning.

    This must be called for a character before using 'tts' or 'tts_async'.

    Args:
        character_name (str): The name of the character.
        audio_path (str | PathLike): The file path to the reference audio (e.g., a WAV file).
        audio_text (str): The transcript of the reference audio.
        language (str): The language of the reference audio.
    """
    audio_path: str = os.fspath(audio_path)

    # 检查文件后缀是否支持
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        logger.error(
            f"Audio format '{ext}' is not supported. Only the following formats are supported: {SUPPORTED_AUDIO_EXTS}"
        )
        return

    if language is None:
        gsv_model = model_manager.get(character_name)
        if gsv_model:
            language = gsv_model.LANGUAGE
        else:
            raise ValueError('No language specified')
    language = normalize_language(language)
    if language not in ['Japanese', 'English', 'Chinese', 'Hybrid-Chinese-English', 'Korean']:
        raise ValueError('Unknown language')

    _reference_audios[character_name] = {
        'audio_path': audio_path,
        'audio_text': audio_text,
        'language': language,
    }
    # print(_reference_audios[character_name])
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=audio_path,
        prompt_text=audio_text,
        language=language,
    )


async def tts_async(
        character_name: str,
        text: str,
        play: bool = False,
        split_sentence: bool = False,
        save_path: Union[str, PathLike, None] = None,
) -> AsyncIterator[bytes]:
    """
    Asynchronously generates speech from text and yields audio chunks.

    This function returns an async iterator that provides the audio data in
    real-time as it's being generated.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio as it's generated. Defaults to False.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis. Defaults to False.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.

    Yields:
        bytes: A chunk of the generated audio data.

    Raises:
        ValueError: If 'set_reference_audio' has not been called for the character.
    """
    if character_name not in _reference_audios:
        raise ValueError("Please call 'set_reference_audio' first to set the reference audio.")

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # 1. 创建 asyncio 队列和获取当前事件循环
    stream_queue: asyncio.Queue[Union[bytes, None]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # 2. 定义回调函数，用于在线程和 asyncio 之间安全地传递数据
    def tts_chunk_callback(c: Optional[bytes]):
        """This callback is called from the TTS worker thread."""
        loop.call_soon_threadsafe(stream_queue.put_nowait, c)

    # 设置 TTS 上下文
    context.current_speaker = character_name
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=_reference_audios[character_name]['audio_path'],
        prompt_text=_reference_audios[character_name]['audio_text'],
        language=_reference_audios[character_name]['language'],
    )

    # 3. 使用新的回调接口启动 TTS 会话
    tts_player.start_session(
        play=play,
        split=split_sentence,
        save_path=save_path,
        chunk_callback=tts_chunk_callback,
    )

    # 馈送文本并通知会话结束
    tts_player.feed(text)
    tts_player.end_session()

    # 4. 从队列中异步读取数据并产生
    while True:
        chunk = await stream_queue.get()
        if chunk is None:
            break
        yield chunk


def tts(
        character_name: str,
        text: str,
        play: bool = False,
        split_sentence: bool = True,
        save_path: Union[str, PathLike, None] = None,
) -> None:
    """
    Synchronously generates speech from text.

    This is a blocking function that will not return until the entire TTS
    process is complete.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.
    """
    if character_name not in _reference_audios:
        logger.error("Please call 'set_reference_audio' first to set the reference audio.")
        return

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    context.current_speaker = character_name
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=_reference_audios[character_name]['audio_path'],
        prompt_text=_reference_audios[character_name]['audio_text'],
        language=_reference_audios[character_name]['language'],
    )

    tts_player.start_session(
        play=play,
        split=split_sentence,
        save_path=save_path,
    )
    tts_player.feed(text)
    tts_player.end_session()
    tts_player.wait_for_tts_completion()


def wait_for_playback_done() -> None:
    """
    Wait until all TTS tasks have finished processing and playback has fully completed.
    """
    tts_player.wait_for_playback_done()


def stop() -> None:
    """
    Stops the currently playing text-to-speech audio.
    """
    tts_player.stop()


def convert_to_onnx(
        torch_ckpt_path: Union[str, PathLike],
        torch_pth_path: Union[str, PathLike],
        output_dir: Union[str, PathLike],
) -> None:
    """
    Converts PyTorch model checkpoints to the ONNX format.

    This function requires PyTorch to be installed.

    Args:
        torch_ckpt_path (str | PathLike): The path to the T2S model (.ckpt) file.
        torch_pth_path (str | PathLike): The path to the VITS model (.pth) file.
        output_dir (str | PathLike): The directory where the ONNX models will be saved.
    """
    try:
        import torch
    except ImportError:
        logger.error("❌ PyTorch is not installed. Please run `pip install torch` first.")
        return

    from .Converter.Converter import convert

    torch_ckpt_path = os.fspath(torch_ckpt_path)
    torch_pth_path = os.fspath(torch_pth_path)
    output_dir = os.fspath(output_dir)

    convert(
        torch_pth_path=torch_pth_path,
        torch_ckpt_path=torch_ckpt_path,
        output_dir=output_dir,
    )


def clear_reference_audio_cache() -> None:
    """
    Clears the cache of reference audio data.
    """
    ReferenceAudio.clear_cache()


def load_predefined_character(character_name: str) -> None:
    """
    Download and load a predefined character model for TTS inference.
    """
    character_name = character_name.lower().strip()
    if character_name not in CHARA_ALIAS_MAP:
        logger.error(f"No predefined character model found for {character_name}")
        return
    character_name = CHARA_ALIAS_MAP[character_name]

    save_path = download_chara(character_name)
    model_manager.load_character(
        character_name=character_name,
        model_dir=os.path.join(save_path, 'tts_models'),
        language=CHARA_LANG[character_name],
    )

    with open(os.path.join(save_path, "prompt_wav.json"), "r", encoding="utf-8") as f:
        prompt_wav_dict: Dict[str, Dict[str, str]] = json.load(f)

    audio_text = prompt_wav_dict["Normal"]["text"]
    audio_path = os.path.join(save_path, "prompt_wav", prompt_wav_dict["Normal"]["wav"])
    _reference_audios[character_name] = {
        'audio_path': audio_path,
        'audio_text': audio_text,
        'language': CHARA_LANG[character_name],
    }
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=audio_path,
        prompt_text=audio_text,
        language=CHARA_LANG[character_name],
    )
