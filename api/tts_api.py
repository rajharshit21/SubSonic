# Location: backend/api/tts_api.py

from fastapi import APIRouter, Form
from fastapi.responses import FileResponse, JSONResponse
from models.multi_voice_tts import synthesize_speech, get_available_voices
import uuid
import os

router = APIRouter()

OUTPUT_DIR = "temp/tts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.get("/tts/voices")
def list_available_voices():
    """Returns a list of available speaker names from the Coqui TTS model."""
    return {"voices": get_available_voices()}

@router.post("/tts/multi-voice")
async def generate_tts(
    text: str = Form(...),
    voice: str = Form(...)
):
    """Generate TTS audio using selected voice."""
    filename = f"tts_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        synthesize_speech(text=text, speaker=voice, output_path=output_path)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="tts_output.wav"
    )
