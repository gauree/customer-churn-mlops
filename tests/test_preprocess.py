import os
import subprocess
import time
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESS_SCRIPT = os.path.join(REPO_ROOT, "src", "data", "preprocess.py")
TRAIN_CSV = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
TEST_CSV = os.path.join(REPO_ROOT, "data", "processed", "test.csv")

def rm_if_exists(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def test_preprocess_creates_processed_files():
    if os.path.exists(TRAIN_CSV):
        os.remove(TRAIN_CSV)
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)

    assert os.path.exists(PREPROCESS_SCRIPT), f"Preprocess script not found: {PREPROCESS_SCRIPT}"
    proc = subprocess.run(["python", PREPROCESS_SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    time.sleep(1)

    assert proc.returncode == 0, f"Preprocess script failed:\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"

    assert os.path.exists(TRAIN_CSV), f"{TRAIN_CSV} not created."
    assert os.path.exists(TEST_CSV), f"{TEST_CSV} not created."

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    assert df_train.shape[0] > 0, "train.csv is empty"
    assert df_test.shape[0] > 0, "test.csv is empty"
