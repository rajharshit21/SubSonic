from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from audio_engine.effects import autotune
from services.file_handler import save_upload_file
from audio_engine.effects.basic import apply_pitch_and_speed
from audio_engine.effects.clarity import clarity_boost
from audio_engine.effects.denoise import remove_noise
from models.openai_filter import apply_openai_style
from database.session_logger import log_transformation
import librosa
from services.file_handler import ensure_wav
from audio_engine.effects.autotune import autotune_chunk
router = APIRouter()

# ✅ Main route: Upload + all filters + optional OpenAI style
@router.post("/transform/upload")
async def upload_and_process_audio(
    file: UploadFile = File(...),
    pitch_shift: int = Form(0),
    time_stretch: float = Form(1.0),
    clarity: bool = Form(False),
    denoise: bool = Form(False),
    style: str = Form("")
):
    print("Received pitch:", pitch_shift)
    print("Received speed:", time_stretch)
    print("Received clarity:", clarity)
    print("Received denoise:", denoise)
    print("Received style:", style)

    # 1) Save original upload
    raw_path = await save_upload_file(file, "data/raw")
    raw_path = ensure_wav(raw_path)

    # 2) Apply pitch/speed
    processed = apply_pitch_and_speed(raw_path, pitch_shift, time_stretch)

    # 3) Optional clarity / denoise
    if clarity:
        processed = clarity_boost(processed)
    if denoise:
        processed = remove_noise(processed)
    if autotune:
        print("[Autotune] Applying pitch correction")
        y, sr = librosa.load(processed, sr=None)
        y = autotune_chunk(y, sr)
        # Optionally, save the autotuned audio back to a file for further processing
        import soundfile as sf
        sf.write(processed, y, sr)
    # 4) Optional OpenAI style filter
    if style:
        processed = await apply_openai_style(processed, style)

    # 5) Log duration
    duration = librosa.get_duration(path=processed)

    # 6) Write transformation log
    log_transformation(
        file_name=file.filename,
        filters_used=[
            f"pitch:{pitch_shift}",
            f"speed:{time_stretch}",
            f"clarity:{clarity}",
            f"denoise:{denoise}",
            f"style:{style or 'none'}"
        ],
        duration=duration
    )

    # 7) Send processed file
    return FileResponse(
        processed,
        media_type="audio/wav",
        filename="processed.wav"
    )
