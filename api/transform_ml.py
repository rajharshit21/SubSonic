# api/transform_ml.py

import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from pathlib import Path
from models.voice_transfer import transfer_voice

router = APIRouter(prefix="/transform", tags=["Voice Transfer"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/voice-transfer")
async def voice_transfer_route(
    file: UploadFile = File(...),
    target_voice: str = Form(...),
    use_dummy: bool = Form(True)
):
    """
    Upload audio + target voice label → return voice-transferred audio
    """
    input_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{input_id}_{file.filename}"
    output_path = OUTPUT_DIR / f"voice_transfer_{input_id}.wav"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Convert voice using ML model
    try:
        result_path = transfer_voice(str(input_path), target_voice, str(output_path), use_dummy=use_dummy)
    except Exception as e:
        return {"error": f"Voice transfer failed: {str(e)}"}

    return FileResponse(result_path, media_type="audio/wav", filename=output_path.name)
