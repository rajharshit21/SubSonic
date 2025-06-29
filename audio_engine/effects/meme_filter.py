# File: effects/fun_filters.py
  # in meme_filter.py
import librosa
import numpy as np
import soundfile as sf
import os

def apply_fun_filter(input_path: str, effect: str) -> str:
    print(f">> Applying fun filter: {effect}")   # ✅ inside the function

    y, sr = librosa.load(input_path, sr=None)

    if effect == "chipmunk":
        y_mod = librosa.effects.pitch_shift(y, sr, n_steps=8)
    elif effect == "robot":
        y_mod = y * np.sin(2 * np.pi * 30 * np.arange(len(y)) / sr)
    elif effect == "alien":
        y_mod = librosa.effects.pitch_shift(y, sr, n_steps=-6)
    else:
        raise ValueError("Unsupported effect")

    output_path = input_path.replace(".wav", f"_{effect}.wav")
    sf.write(output_path, y_mod, sr)
    return output_path
