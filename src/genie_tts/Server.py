import asyncio
import os
from typing import AsyncIterator, Optional, Callable, Union, Dict
import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .Audio.ReferenceAudio import ReferenceAudio
from .Core.TTSPlayer import tts_player
from .ModelManager import model_manager
from .Utils.Shared import context

logger = logging.getLogger(__name__)

_reference_audios: Dict[str, dict] = {}
SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}

app = FastAPI()


class CharacterPayload(BaseModel):
    character_name: str
    onnx_model_dir: str


class UnloadCharacterPayload(BaseModel):
    character_name: str


class ReferenceAudioPayload(BaseModel):
    character_name: str
    audio_path: str
    audio_text: str


class TTSPayload(BaseModel):
    character_name: str
    text: str
    split_sentence: bool = False
    save_path: Optional[str] = None


@app.post("/load_character")
def load_character_endpoint(payload: CharacterPayload):
    try:
        model_manager.load_character(
            character_name=payload.character_name,
            model_dir=payload.onnx_model_dir,
        )
        return {"status": "success", "message": f"Character '{payload.character_name}' loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload_character")
def unload_character_endpoint(payload: UnloadCharacterPayload):
    try:
        model_manager.remove_character(character_name=payload.character_name)
        return {"status": "success", "message": f"Character '{payload.character_name}' unloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_reference_audio")
def set_reference_audio_endpoint(payload: ReferenceAudioPayload):
    ext = os.path.splitext(payload.audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio format '{ext}' is not supported. Supported formats: {SUPPORTED_AUDIO_EXTS}",
        )
    _reference_audios[payload.character_name] = {
        'audio_path': payload.audio_path,
        'audio_text': payload.audio_text,
    }
    return {"status": "success", "message": f"Reference audio for '{payload.character_name}' set."}


def run_tts_in_background(
        character_name: str,
        text: str,
        split_sentence: bool,
        save_path: Optional[str],
        chunk_callback: Callable[[Optional[bytes]], None]
):
    try:
        context.current_speaker = character_name
        context.current_prompt_audio = ReferenceAudio(
            prompt_wav=_reference_audios[character_name]['audio_path'],
            prompt_text=_reference_audios[character_name]['audio_text'],
        )
        tts_player.start_session(
            play=False,
            split=split_sentence,
            save_path=save_path,
            chunk_callback=chunk_callback,
        )
        tts_player.feed(text)
        tts_player.end_session()
        tts_player.wait_for_tts_completion()
    except Exception as e:
        logger.error(f"Error in TTS background task: {e}", exc_info=True)


async def audio_stream_generator(queue: asyncio.Queue) -> AsyncIterator[bytes]:
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/tts")
async def tts_endpoint(payload: TTSPayload):
    if payload.character_name not in _reference_audios:
        raise HTTPException(status_code=404, detail="Character not found or reference audio not set.")

    loop = asyncio.get_running_loop()
    stream_queue: asyncio.Queue[Union[bytes, None]] = asyncio.Queue()

    def tts_chunk_callback(chunk: Optional[bytes]):
        loop.call_soon_threadsafe(stream_queue.put_nowait, chunk)

    loop.run_in_executor(
        None,
        run_tts_in_background,
        payload.character_name,
        payload.text,
        payload.split_sentence,
        payload.save_path,
        tts_chunk_callback
    )

    return StreamingResponse(audio_stream_generator(stream_queue), media_type="audio/wav")


@app.post("/stop")
def stop_endpoint():
    try:
        tts_player.stop()
        return {"status": "success", "message": "TTS stopped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_reference_audio_cache")
def clear_reference_audio_cache_endpoint():
    try:
        ReferenceAudio.clear_cache()
        return {"status": "success", "message": "Reference audio cache cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "127.0.0.1", port: int = 8000, workers: int = 1):
    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    start_server(host="0.0.0.0", port=8000, workers=1)
