# main.py
import os
import sys
import shutil
import tempfile
import gc
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from pydub import AudioSegment

# Local imports
from api.routes import router as audio_router
from api.live_audio_ws import router as live_router
from api import analyze as analytics
from api import tts_api

# === App Init ===
app = FastAPI()

# === CORS (for Vercel <-> Render) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://subsonic.vercel.app",  # frontend on Vercel
        "http://localhost:5173",        # local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Env Vars ===
load_dotenv()

# === FFMPEG Setup ===
AudioSegment.converter = shutil.which("ffmpeg")

# === Temp Directory ===
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# === Routers ===
app.include_router(audio_router, prefix="/api")
app.include_router(live_router)
app.include_router(analytics.router, prefix="/api")
app.include_router(tts_api.router, prefix="/api")

# =========================
# Root Endpoint
# =========================
@app.get("/")
def read_root():
    return {"message": "Welcome to SubSonic Voice Changer API"}

# =========================
# WebSocket: live mic input
# =========================
@app.websocket("/ws/live")
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
# Audio Transform Endpoint
# =========================
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
    tmp_path = None
    try:
        # 1️⃣ Save uploaded file to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2️⃣ Load + downsample to save memory
        sound = AudioSegment.from_file(tmp_path, format="mp3")
        sound = sound.set_channels(1).set_frame_rate(16000)

        # 3️⃣ Pitch shift
        if pitch_shift != 0:
            sound = sound._spawn(
                sound.raw_data,
                overrides={"frame_rate": int(sound.frame_rate * (2.0 ** (pitch_shift / 12.0)))}
            ).set_frame_rate(16000)

        # 4️⃣ Time stretch (speed change)
        if time_stretch != 1.0:
            sound = sound._spawn(
                sound.raw_data,
                overrides={"frame_rate": int(sound.frame_rate * time_stretch)}
            ).set_frame_rate(16000)

        # TODO: clarity, denoise, autotune, style → add back later (memory heavy)

        # 5️⃣ Export result
        out_file = tmp_path.replace(".mp3", "_out.mp3")
        sound.export(out_file, format="mp3")

        # Free memory
        del sound
        gc.collect()

        return FileResponse(out_file, filename="output.mp3", media_type="audio/mpeg")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
