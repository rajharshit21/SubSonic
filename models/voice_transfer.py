# models/voice_transfer.py
"""
Single entry‑point: transfer_voice()

If use_dummy=True it just copies the file (fast local test).
Otherwise it calls So‑VITS‑SVC’s inference_main.py to convert the voice.
"""

import shutil
import subprocess
from pathlib import Path

# ------------------------------------------------------------------
# 🔧 Configure where the So‑VITS‑SVC model + config live
# ------------------------------------------------------------------
SVC_DIR   = Path(__file__).parent / "so-vits-svc4.1-pretrain_model"  # adjust if you moved the folder
CONFIG    = SVC_DIR / "config.json"
MODEL     = SVC_DIR / "G_10000.pth"        # or whatever checkpoint you downloaded
CLUSTER   = ""                             # leave blank if you don’t use a cluster model

# Path to inference_main.py (comes from the repo you cloned)
INFERENCE_SCRIPT = Path(__file__).parent / "inference_main.py"
# If inference_main.py is on your PYTHONPATH, you can just use "inference_main.py"
# INFERENCE_SCRIPT = "inference_main.py"

# ------------------------------------------------------------------
def transfer_voice(
    input_path: str,
    target_voice: str,
    output_path: str,
    use_dummy: bool = False
) -> str:
    """
    Convert `input_path` to `target_voice`, save as `output_path`, return the path.
    """
    if use_dummy:
        # Fast path for dev / unit tests
        shutil.copy(input_path, output_path)
        return output_path

    # Build So‑VITS‑SVC CLI command
    cmd = [
        "python", str(INFERENCE_SCRIPT),
        "-i", input_path,
        "-o", output_path,
        "-s", target_voice,
        "--config", str(CONFIG),
        "--model",  str(MODEL),
        "--cluster_model_path", str(CLUSTER)
    ]

    # Run and raise an error if the model fails
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Voice transfer failed:\n{err}") from err

    return output_path
