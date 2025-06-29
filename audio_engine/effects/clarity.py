# audio_engine/effects/clarity.py

from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

# ────────────────────────────────────────────────────────
# CONFIG
DEFAULT_CUTOFF = 100.0  # Hz
DEFAULT_ORDER = 5
PCM_MAX = 32767

# ────────────────────────────────────────────────────────
# HELPERS

def highpass_filter(data, cutoff=DEFAULT_CUTOFF, fs=16000, order=DEFAULT_ORDER):
    """Apply a high-pass Butterworth filter to remove low frequencies."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, data)


# ────────────────────────────────────────────────────────
# FILE-BASED VERSION

def clarity_boost(input_path: str, output_path: str | None = None) -> str:
    """
    Apply clarity EQ (high-pass + normalize) to a WAV file.
    Returns output file path.
    """
    print(">> Applying clarity filter (file mode)")

    try:
        y, sr = librosa.load(input_path, sr=None)
        if y.size == 0:
            raise ValueError("Empty audio signal")

        y_hp = highpass_filter(y, cutoff=DEFAULT_CUTOFF, fs=sr)
        y_norm = librosa.util.normalize(y_hp)

        if not output_path:
            output_path = input_path.replace(".wav", "_clarity.wav")

        sf.write(output_path, y_norm, sr)
        return output_path

    except Exception as e:
        print(f"[clarity_boost] ERROR: {e}")
        raise


# ────────────────────────────────────────────────────────
# CHUNK-BASED VERSION

def clarity_boost_chunk(data: bytes, frame_rate: int = 16000) -> bytes:
    """
    Apply clarity boost to raw 16-bit mono PCM chunk.
    Returns processed PCM bytes.
    """
    try:
        seg = AudioSegment(
            data=data,
            sample_width=2,
            frame_rate=frame_rate,
            channels=1
        )
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)

        if samples.size == 0:
            return data  # skip processing for empty chunks

        y_hp = highpass_filter(samples, cutoff=DEFAULT_CUTOFF, fs=frame_rate)
        y_norm = librosa.util.normalize(y_hp)

        int16 = np.clip(y_norm * PCM_MAX, -PCM_MAX, PCM_MAX).astype(np.int16)
        out_seg = AudioSegment(
            int16.tobytes(),
            frame_rate=frame_rate,
            sample_width=2,
            channels=1
        )
        return out_seg.raw_data

    except Exception as e:
        print(f"[clarity_boost_chunk] ERROR: {e}")
        return data
