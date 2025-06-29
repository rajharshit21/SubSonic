# services/file_handler.py

import shutil
from pathlib import Path
from fastapi import UploadFile

from pydub import AudioSegment
import uuid, os
RAW_AUDIO_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def get_filename(path: str) -> str:
    return Path(path).stem


def ensure_dirs():
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload_file(file: UploadFile, folder: str = "data/raw") -> str:
    """Save an uploaded file to the specified folder with a unique name."""
    target_dir = Path(folder)
    target_dir.mkdir(parents=True, exist_ok=True)

    unique_suffix = uuid.uuid4().hex[:8]
    filename = f"{Path(file.filename).stem}_{unique_suffix}{Path(file.filename).suffix}"
    filepath = target_dir / filename

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(filepath)


def ensure_wav(in_path: str) -> str:
    """If input is not .wav, convert to wav and return new path."""
    if in_path.lower().endswith(".wav"):
        return in_path
    out_path = os.path.splitext(in_path)[0] + f"_{uuid.uuid4().hex[:6]}.wav"
    AudioSegment.from_file(in_path).export(out_path, format="wav")
    return out_path