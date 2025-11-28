import os
import subprocess
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "src", "model", "train.py")
MODEL_PATH = os.path.join(REPO_ROOT, "src", "model", "model.pkl")

def test_train_script_creates_model():
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

    assert os.path.exists(TRAIN_SCRIPT), f"Train script not found: {TRAIN_SCRIPT}"

    proc = subprocess.run(["python", TRAIN_SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    time.sleep(1)

    assert proc.returncode == 0, f"Train script failed with return code {proc.returncode}\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    assert os.path.exists(MODEL_PATH), f"Expected model not found at {MODEL_PATH}"
