# audio_engine/effects/autotune.py

import numpy as np
import librosa
import soundfile as sf
import tempfile
import subprocess
from scipy.interpolate import interp1d
from pathlib import Path


def estimate_and_interpolate_f0(y, sr):
    """ Estimate fundamental frequency and interpolate unvoiced frames. """
    f0, voiced_flag, _ = librosa.pyin(y,
                                      fmin=librosa.note_to_hz('C2'),
                                      fmax=librosa.note_to_hz('C7'))

    # Interpolate missing (unvoiced) values
    indices = np.arange(len(f0))
    voiced_idx = indices[~np.isnan(f0)]
    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) < 2:
        return None  # too little pitch info

    interp_fn = interp1d(voiced_idx, voiced_f0, bounds_error=False, fill_value="extrapolate")
    return interp_fn(indices)


def snap_f0_to_scale(f0, scale='C'):
    """ Snap each f0 to nearest note in the chosen scale. """
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    scale_freqs = [librosa.note_to_hz(f"{n}4") for n in notes]

    snapped = np.array([min(scale_freqs, key=lambda x: abs(x - pitch)) if pitch > 0 else pitch
                        for pitch in f0])
    return snapped


def autotune_chunk(y, sr, scale='C'):
    """ Autotune signal by estimating pitch and snapping to musical scale. """
    print(">> [Autotune] Starting studio-grade processing")

    f0 = estimate_and_interpolate_f0(y, sr)
    if f0 is None:
        print("[Autotune] Skipping autotune (insufficient voiced signal)")
        return y

    f0_target = snap_f0_to_scale(f0, scale)
    semitone_shift = 12 * np.log2(f0_target / f0)
    semitone_shift = np.clip(semitone_shift, -12, 12)  # prevent extreme shifts

    avg_shift = np.nanmean(semitone_shift)
    print(f"[Autotune] Avg semitone shift: {avg_shift:.2f}")

    # === Rubberband CLI for high-quality pitch shift ===
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.wav"
        out_path = Path(tmpdir) / "output.wav"

        sf.write(in_path, y, sr)

        cmd = [
            "rubberband",
            "-p", f"{avg_shift:.2f}",
            "-c",  # preserve duration
            str(in_path),
            str(out_path)
        ]

        try:
            subprocess.run(cmd, check=True)
            y_out, _ = librosa.load(out_path, sr=sr)
            return y_out
        except Exception as e:
            print(f"[Autotune ERROR] {e}")
            return y  # fallback to original
