import os
import joblib
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "src", "model", "model.pkl")
TRAIN_CSV = os.path.join(REPO_ROOT, "data", "processed", "train.csv")

def build_valid_input_row():
    """
    Return a single-row pandas DataFrame compatible with the model.
    Strategy:
      1) If processed train CSV exists, take its first row and drop a likely target column.
      2) If not available, raise a clear error so the environment can be fixed.
    """
    if not os.path.exists(TRAIN_CSV):
        raise RuntimeError(f"Processed train CSV not found at {TRAIN_CSV}. Run preprocessing first.")

    df = pd.read_csv(TRAIN_CSV, nrows=1)
    possible_targets = {"Churn", "target", "label", "y", "is_churn"}
    drop_cols = [c for c in df.columns if c in possible_targets]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    else:
        last_col = df.columns[-1]
        if not pd.api.types.is_numeric_dtype(df[last_col]):
            df = df.drop(columns=[last_col], errors="ignore")

    return df

def test_model_file_exists_and_has_predict():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"

    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model has no 'predict' method"
    X_df = build_valid_input_row()
    pred = model.predict(X_df)
    assert hasattr(pred, "__len__") and len(pred) == 1
