# Location: models/deep_denoise.py

import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import tempfile
import os

def deep_denoise(input_path: str) -> str:
    print("[Deep Denoise] Loading and processing audio")
    y, sr = librosa.load(input_path, sr=16000)

    # Estimate noise from first 0.5 seconds
    noise_sample = y[:sr // 2]

    reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=1.0)

    # Save to temporary output path
    _, temp_path = tempfile.mkstemp(suffix=".wav")
    sf.write(temp_path, reduced, sr)
    return temp_path
