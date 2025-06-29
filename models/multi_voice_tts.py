import os
import requests

# Determine if we're in Google Colab
try:
    import importlib
    google_colab = importlib.import_module("google.colab")
    userdata = importlib.import_module("google.colab.userdata")
    TTS_BACKEND_URL = userdata.get("TTS_BACKEND_URL")
except Exception:
    TTS_BACKEND_URL = os.getenv("TTS_BACKEND_URL")


# Available speakers (fixed list, since we removed local TTS)
AVAILABLE_SPEAKERS = ["narrator", "female", "male", "robot", "custom"]

def get_available_voices():
    return AVAILABLE_SPEAKERS

def synthesize_speech(text: str, speaker: str, output_path: str):
    if not TTS_BACKEND_URL:
        raise RuntimeError("TTS_BACKEND_URL is not set. Cannot use remote TTS.")

    if speaker not in AVAILABLE_SPEAKERS:
        raise ValueError(f"Speaker '{speaker}' is invalid. Choose from {AVAILABLE_SPEAKERS}.")

    response = requests.post(
        TTS_BACKEND_URL,
        json={"text": text, "speaker": speaker},
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Remote TTS failed: {response.status_code} {response.text}")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
