# Location: backend/api/live_audio_ws.py

import io
import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, Query
from starlette.websockets import WebSocketState

# ── Chunk‑level helpers (add these tiny functions in your effects modules) ──
from audio_engine.effects.basic import pitch_speed_chunk       # new
from audio_engine.effects.clarity import clarity_boost_chunk   # new
from audio_engine.effects.denoise import remove_noise_chunk     # new

router = APIRouter()

TARGET_SR   = 16_000        # 16‑kHz mono
FRAME_SIZE  = 2048          # must match ScriptProcessorNode in JS

# ── Utility converters ─────────────────────────────────────────────────────
def int16_to_float32(buf: bytes) -> np.ndarray:
    audio_i16 = np.frombuffer(buf, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0)

def float32_to_wav_bytes(audio: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Return WAV‑encoded bytes for a float32 [-1,1] numpy signal."""
    out = io.BytesIO()
    sf.write(out, audio, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()

# ───────────────────────────────────────────────────────────────────────────
@router.websocket("/ws/audio")
async def websocket_audio_stream(
    websocket: WebSocket,
    clarity: bool = Query(False),
    denoise: bool = Query(False),
    pitch: int   = Query(0),
    speed: float = Query(1.0)
):
    """Bidirectional real‑time audio: receives raw PCM int16, sends back filtered WAV bytes."""
    await websocket.accept()

    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            raw_pcm: bytes = await websocket.receive_bytes()     # ← 2048‑frame int16 buffer
            audio = int16_to_float32(raw_pcm)

            # ── Apply chosen effects, frame‑wise ───────────────────────────
            if pitch or speed != 1.0:
                audio = pitch_speed_chunk(audio, TARGET_SR, pitch, speed)

            if clarity:
                audio = clarity_boost_chunk(audio, TARGET_SR)

            if denoise:
                audio = remove_noise_chunk(audio, TARGET_SR)
            # ───────────────────────────────────────────────────────────────

            wav_bytes = float32_to_wav_bytes(audio, TARGET_SR)
            await websocket.send_bytes(wav_bytes)

    except Exception as exc:
        print("[WebSocket closed]", exc)
    finally:
        await websocket.close()
