# 文件: .../Core/TTSPlayer.py

import queue
import os
import threading

import numpy as np
import wave
from typing import Optional, List, Callable
import logging

from ..Utils.TextSplitter import TextSplitter
from ..Core.Inference import tts_client
from ..ModelManager import model_manager
from ..Utils.Shared import context
from ..Utils.Utils import clear_queue

logger = logging.getLogger(__name__)

STREAM_END = 'STREAM_END'  # 这是一个特殊的标记，表示文本流结束
AUDIO_STREAM_END = 'AUDIO_STREAM_END'  # 新增：特殊的标记，表示音频流播放结束


class TTSPlayer:
    def __init__(self, sample_rate: int = 32000):
        self._text_splitter = TextSplitter()

        self.sample_rate: int = sample_rate
        self.channels: int = 1
        self.bytes_per_sample: int = 2  # 16-bit audio

        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._stop_event: threading.Event = threading.Event()
        self._tts_done_event: threading.Event = threading.Event()
        self._playback_done_event: threading.Event = threading.Event()  # 新增：用于标记播放完成
        self._api_lock: threading.Lock = threading.Lock()

        self._tts_worker: Optional[threading.Thread] = None
        self._playback_worker: Optional[threading.Thread] = None

        self._play: bool = False
        self._current_save_path: Optional[str] = None
        self._session_audio_chunks: List[np.ndarray] = []
        self._split: bool = False

        self._chunk_callback: Optional[Callable[[Optional[bytes]], None]] = None

    @staticmethod
    def _preprocess_for_playback(audio_float: np.ndarray) -> bytes:
        audio_int16 = (audio_float.squeeze() * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _tts_worker_loop(self):
        """从文本队列取句子，生成音频，并通过回调函数或音频队列分发。"""
        while not self._stop_event.is_set():
            try:
                sentence = self._text_queue.get(timeout=1)
                if sentence is None or self._stop_event.is_set():
                    break
            except queue.Empty:
                continue

            try:
                if sentence is STREAM_END:
                    if self._current_save_path and self._session_audio_chunks:
                        self._save_session_audio()

                    # 在TTS工作线程完成时，通过回调发送结束信号
                    if self._chunk_callback:
                        self._chunk_callback(None)

                    # 新增：如果开启了播放，通知音频队列流已结束
                    if self._play:
                        self._audio_queue.put(AUDIO_STREAM_END)

                    self._tts_done_event.set()
                    continue

                gsv_model = model_manager.get(context.current_speaker)
                if not gsv_model or not context.current_prompt_audio:
                    logger.error("Missing model or reference audio.")
                    continue

                tts_client.stop_event.clear()
                audio_chunk = tts_client.tts(
                    text=sentence,
                    prompt_audio=context.current_prompt_audio,
                    encoder=gsv_model.T2S_ENCODER,
                    first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                    stage_decoder=gsv_model.T2S_STAGE_DECODER,
                    vocoder=gsv_model.VITS,
                    prompt_encoder=gsv_model.PROMPT_ENCODER,
                    language=gsv_model.LANGUAGE,
                )

                if audio_chunk is not None:
                    if self._play:
                        self._audio_queue.put(audio_chunk)
                    if self._current_save_path:
                        self._session_audio_chunks.append(audio_chunk)

                    # 使用回调函数处理流式数据
                    if self._chunk_callback:
                        audio_data = self._preprocess_for_playback(audio_chunk)
                        self._chunk_callback(audio_data)

            except Exception as e:
                logger.error(f"A critical error occurred while processing the TTS task: {e}", exc_info=True)
                # 发生错误时，也要确保发送结束信号
                if self._chunk_callback:
                    self._chunk_callback(None)
                self._tts_done_event.set()

    def _playback_worker_loop(self):
        try:
            import sounddevice as sd
            with sd.OutputStream(samplerate=self.sample_rate,
                                 channels=self.channels,
                                 dtype='float32') as stream:
                while not self._stop_event.is_set():
                    try:
                        audio_chunk = self._audio_queue.get(timeout=1)
                        if audio_chunk is None:
                            break
                        if audio_chunk is AUDIO_STREAM_END:
                            self._playback_done_event.set()
                            continue
                        stream.write(audio_chunk.squeeze())
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error during audio playback: {e}", exc_info=True)

        except Exception as e:
            logger.warning(f"Failed to initialize sounddevice: {e}. Audio playback will be skipped.")
            # 如果音频设备初始化失败，即使不播放，也要消费队列中的结束信号，防止主线程死锁
            while not self._stop_event.is_set():
                try:
                    item = self._audio_queue.get(timeout=0.5)
                    if item is None:
                        break
                    if item is AUDIO_STREAM_END:
                        self._playback_done_event.set()
                except queue.Empty:
                    continue

    def _save_session_audio(self):
        try:
            full_audio = np.concatenate(self._session_audio_chunks, axis=0)
            with wave.open(self._current_save_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.bytes_per_sample)
                wf.setframerate(self.sample_rate)
                wf.writeframes(self._preprocess_for_playback(full_audio))
            logger.info(f"Audio successfully saved to {os.path.abspath(self._current_save_path)}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
        finally:
            self._session_audio_chunks = []
            self._current_save_path = None

    def start_session(
            self,
            play: bool = False,
            split: bool = False,
            save_path: Optional[str] = None,
            chunk_callback: Optional[Callable[[Optional[bytes]], None]] = None
    ):
        with self._api_lock:
            self._tts_done_event.clear()
            self._playback_done_event.clear()  # 新增：重置播放完成事件
            self._chunk_callback = chunk_callback
            self._stop_event.clear()

            if self._tts_worker is None or not self._tts_worker.is_alive():
                self._tts_worker = threading.Thread(target=self._tts_worker_loop, daemon=True)
                self._tts_worker.start()

            if self._playback_worker is None or not self._playback_worker.is_alive():
                self._playback_worker = threading.Thread(target=self._playback_worker_loop, daemon=True)
                self._playback_worker.start()

            clear_queue(self._text_queue)
            clear_queue(self._audio_queue)

            self._play = play
            self._split = split
            self._current_save_path = save_path
            self._session_audio_chunks = []

    def feed(self, text_chunk: str):
        with self._api_lock:
            if not text_chunk:
                return
            if self._split:
                sentences = self._text_splitter.split(text_chunk.strip())
                for sentence in sentences:
                    self._text_queue.put(sentence)
            else:
                self._text_queue.put(text_chunk)

    def end_session(self):
        with self._api_lock:
            self._text_queue.put(STREAM_END)

    def stop(self):
        with self._api_lock:
            if self._tts_worker is None and self._playback_worker is None:
                return
            if self._stop_event.is_set():
                return
            tts_client.stop_event.set()
            self._stop_event.set()
            self._tts_done_event.set()
            self._text_queue.put(None)
            self._audio_queue.put(None)
            if self._tts_worker and self._tts_worker.is_alive():
                self._tts_worker.join()
            if self._playback_worker and self._playback_worker.is_alive():
                self._playback_worker.join()
            self._tts_worker = None
            self._playback_worker = None

    def wait_for_tts_completion(self):
        if self._tts_done_event.is_set():
            return
        self._tts_done_event.wait()

    def wait_for_playback_done(self):
        # 1. 首先等待TTS生成全部完成
        self.wait_for_tts_completion()

        # 2. 如果开启了播放且没有被强制停止，则等待播放结束
        if self._play and not self._stop_event.is_set():
            if not self._playback_done_event.is_set():
                self._playback_done_event.wait()


tts_player: TTSPlayer = TTSPlayer()
