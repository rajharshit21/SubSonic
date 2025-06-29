from database.session_logger import log_transformation
from audio_engine.effects.basic import apply_pitch_and_speed
from audio_engine.effects.meme_filter import apply_fun_filter  
from fastapi import APIRouter, UploadFile, Form
from models.openai_filter import apply_openai_style
import shutil


# Example usage
log_transformation(
    file_name="user_123_pitchfast.wav",
    filters_used=["pitch_shift", "speed_up"],
    duration=3.6
)




router = APIRouter()

@router.post("/transform/openai-style")
async def transform_openai_style(file: UploadFile, style: str = Form(...)):
    input_path = f"data/raw/{file.filename}"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = await apply_openai_style(input_path, style)

    return {"message": "Voice styled successfully", "output_file": output_path}
