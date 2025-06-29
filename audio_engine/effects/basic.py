import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pedalboard import Pedalboard, PitchShift
from io import BytesIO

# ⏺️ FILE-BASED
def apply_pitch_and_speed(input_path, pitch_shift=0, time_stretch=1.0):
    print(f">> Applying pitch {pitch_shift}, speed {time_stretch}")
    y, sr = librosa.load(input_path, sr=None)

    # Time Stretch (first)
    if time_stretch != 1.0:
        if len(y) > 2048:
            y = librosa.effects.time_stretch(y, rate=time_stretch)
        else:
            print("[WARN] Clip too short for time-stretching.")

    # Pitch Shift (via Pedalboard)
    if pitch_shift != 0:
        board = Pedalboard([PitchShift(semitones=pitch_shift)])
        y = board.process(y[np.newaxis, :], sr)[0]  # mono

    output_path = input_path.replace("raw", "processed").replace(".wav", "_processed.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y, sr)
    return output_path

# 🔁 CHUNK-BASED (WebSocket)
def pitch_speed_chunk(data: bytes, frame_rate: int = 16000, pitch: int = 0, speed: float = 1.0) -> bytes:
    chunk = AudioSegment(
        data=data,
        sample_width=2,
        frame_rate=frame_rate,
        channels=1
    )

    samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
    if chunk.channels > 1:
        samples = samples.reshape((-1, chunk.channels)).T
    else:
        samples = samples.reshape((1, -1))

    # Normalize
    samples = samples / np.max(np.abs(samples), initial=1)

    # Pitch shift
    if pitch != 0:
        board = Pedalboard([PitchShift(semitones=pitch)])
        samples = board(samples, sample_rate=frame_rate)

    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767).astype(np.int16).flatten()

    return AudioSegment(
        int16.tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=1
    ).raw_data
