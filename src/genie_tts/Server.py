import asyncio
from typing import AsyncIterator, Optional, Union
import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import Internal

logger = logging.getLogger(__name__)

app = FastAPI()


class CharacterPayload(BaseModel):
    character_name: str
    onnx_model_dir: str
    language: str


class UnloadCharacterPayload(BaseModel):
    character_name: str


class ReferenceAudioPayload(BaseModel):
    character_name: str
    audio_path: str
    audio_text: str
    language: str


class TTSPayload(BaseModel):
    character_name: str
    text: str
    split_sentence: bool = False
    save_path: Optional[str] = None


@app.post("/load_character")
def load_character_endpoint(payload: CharacterPayload):
    try:
        Internal.load_character(
            character_name=payload.character_name,
            onnx_model_dir=payload.onnx_model_dir,
            language=payload.language,
        )
        return {"status": "success", "message": f"Character '{payload.character_name}' loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload_character")
def unload_character_endpoint(payload: UnloadCharacterPayload):
    try:
        Internal.unload_character(character_name=payload.character_name)
        return {"status": "success", "message": f"Character '{payload.character_name}' unloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_reference_audio")
def set_reference_audio_endpoint(payload: ReferenceAudioPayload):
    try:
        Internal.set_reference_audio(
            character_name=payload.character_name,
            audio_path=payload.audio_path,
            audio_text=payload.audio_text,
            language=payload.language,
        )
        return {"status": "success", "message": f"Reference audio for '{payload.character_name}' set."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def audio_stream_generator(queue: asyncio.Queue) -> AsyncIterator[bytes]:
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/tts")
async def tts_endpoint(payload: TTSPayload):
    loop = asyncio.get_running_loop()
    stream_queue: asyncio.Queue[Union[bytes, None]] = asyncio.Queue()

    def tts_chunk_callback(chunk: Optional[bytes]):
        loop.call_soon_threadsafe(stream_queue.put_nowait, chunk)

    loop.run_in_executor(
        None,
        _run_tts_in_background,
        payload.character_name,
        payload.text,
        payload.split_sentence,
        payload.save_path,
        tts_chunk_callback,
    )

    return StreamingResponse(audio_stream_generator(stream_queue), media_type="audio/wav")


def _run_tts_in_background(
        character_name: str,
        text: str,
        split_sentence: bool,
        save_path: Optional[str],
        chunk_callback,
):
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def run():
            async for chunk in Internal.tts_async(
                character_name=character_name,
                text=text,
                play=False,
                split_sentence=split_sentence,
                save_path=save_path,
            ):
                chunk_callback(chunk)
        loop.run_until_complete(run())
    except ValueError as e:
        logger.error(f"TTS error: {e}")
    except Exception as e:
        logger.error(f"Error in TTS background task: {e}", exc_info=True)
    finally:
        chunk_callback(None)


@app.post("/stop")
def stop_endpoint():
    try:
        Internal.stop()
        return {"status": "success", "message": "TTS stopped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_reference_audio_cache")
def clear_reference_audio_cache_endpoint():
    try:
        Internal.clear_reference_audio_cache()
        return {"status": "success", "message": "Reference audio cache cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "127.0.0.1", port: int = 8000, workers: int = 1):
    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    start_server(host="0.0.0.0", port=8000, workers=1)
