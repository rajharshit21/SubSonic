# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydub import AudioSegment

# Local modules
from api.routes import router as audio_router
from api.live_audio_ws import router as live_router
from api import analyze as analytics
from api import tts_api
from audio_engine.effects.denoise import remove_noise
from api.tts_api import router as tts_router
from audio_engine.effects.autotune import autotune_chunk
from models.deep_denoise import deep_denoise
from audio_engine.effects.basic import apply_pitch_and_speed
from audio_engine.effects.clarity import clarity_boost

# === Initialize FastAPI app
app = FastAPI()

# === CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load .env
load_dotenv()

# === Fix for ffmpeg
AudioSegment.converter = shutil.which("ffmpeg")

# === Temp Directory
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# === Routers
app.include_router(audio_router, prefix="/api")
app.include_router(live_router)
app.include_router(analytics.router)
app.include_router(tts_api.router)

# =========================
# WebSocket: live mic input
# =========================
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"Received audio chunk: {len(data)} bytes")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# =========================
# REST Endpoints
# =========================

@app.get("/")
def read_root():
    return {"message": "Welcome to SubSonic Voice Changer API"}




@app.post("/apply_clarity_boost")
async def apply_clarity(file: UploadFile = File(...)):
    input_path = TEMP_DIR / file.filename
    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    output_path = clarity_boost(input_path)
    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)


@app.post("/apply_noise_removal")
async def apply_noise_removal(file: UploadFile = File(...)):
    input_path = TEMP_DIR / file.filename
    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    output_path = remove_noise(input_path)
    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)


@app.post("/api/transform/upload")
async def transform_audio(
    file: UploadFile = File(...),
    pitch_shift: int = Form(0),
    time_stretch: float = Form(1.0),
    clarity: bool = Form(False),
    denoise: bool = Form(False),
    style: str = Form(""),
    autotune: bool = Form(False),
):
    import librosa, soundfile as sf
    input_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"input_{input_id}.wav"
    output_path = TEMP_DIR / f"output_{input_id}.wav"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # === Apply Deep Denoise if enabled ===
    if denoise:
        print("[INFO] Applying deep denoise")
        input_path = Path(deep_denoise(str(input_path)))

    # === Load Audio ===
    y, sr = librosa.load(input_path, sr=16000)

    # === Optional: Autotune ===
    if autotune:
        print("[INFO] Applying autotune")
        y = autotune_chunk(y, sr)

    # === Optional: Clarity ===
    if clarity:
        print("[INFO] Applying clarity boost")
        # apply clarity directly to numpy array instead of via saved file
        from audio_engine.effects.clarity import highpass_filter
        y = highpass_filter(y, cutoff=100.0, fs=sr)
        y = librosa.util.normalize(y)

    # === Optional: Pitch + Speed ===
    if pitch_shift != 0 or time_stretch != 1.0:
        print("[INFO] Applying pitch & speed")
        y = apply_pitch_and_speed(y, sr, pitch_shift, time_stretch)

    # === Final output normalization & save ===
    y = librosa.util.normalize(y)
    sf.write(output_path, y, sr)

    return FileResponse(output_path, media_type="audio/wav", filename="processed.wav")
