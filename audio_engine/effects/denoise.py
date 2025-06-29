print(">> Studio Denoise module loaded")

import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment

# ⏺️ File-based studio denoise
def remove_noise(input_path: str, output_path: str = None) -> str:
    print(">> Applying denoise filter (file mode)")
    y, sr = librosa.load(input_path, sr=None)

    y_denoised = nr.reduce_noise(y=y, sr=sr)  # ✅ no use_tensorflow
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised.wav"

    sf.write(output_path, y_denoised, sr)
    return output_path



# 🔁 Real-time chunk-based denoise
def remove_noise_chunk(data: bytes, frame_rate: int = 16000) -> bytes:
    """
    Applies noise reduction to raw audio chunk in bytes.
    Ideal for WebSocket live streaming.
    """
    seg = AudioSegment(
        data=data,
        sample_width=2,
        frame_rate=frame_rate,
        channels=1
    )
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)

    # Normalize for safety
    if np.max(np.abs(samples)) > 0:
        samples /= np.max(np.abs(samples))

    # Apply basic percentile-based noise gate
    noise_floor = np.percentile(np.abs(samples), 10)
    samples[np.abs(samples) < noise_floor] = 0

    int16 = np.int16(samples * 32767)
    out_seg = AudioSegment(
        int16.tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=1
    )
    return out_seg.raw_data
