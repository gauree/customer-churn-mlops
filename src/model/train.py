import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_data():
    root = get_root()
    train_path = os.path.join(root, "data", "processed", "train.csv")
    test_path = os.path.join(root, "data", "processed", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def encode_target(series):
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes", "1", "true"]).astype(int)

def train():
    train_df, test_df = load_data()

    target = "Churn"
    if target not in train_df.columns:
        raise ValueError("Churn column not found in processed data")

    y_train = encode_target(train_df[target])
    X_train = train_df.drop(columns=[target])

    y_test = encode_target(test_df[target])
    X_test = test_df.drop(columns=[target])

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    models = {
        "log_reg": LogisticRegression(max_iter=200),
        "random_forest": RandomForestClassifier(
            n_estimators=120, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=20,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    best_name = None
    best_score = -1.0
    best_pipeline = None

    mlflow.set_experiment("customer_churn_training")

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        with mlflow.start_run(run_name=name):
            pipe.fit(X_tr, y_tr)
            val_acc = pipe.score(X_val, y_val)
            test_acc = pipe.score(X_test, y_test)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.sklearn.log_model(pipe, "model")

        if val_acc > best_score:
            best_score = val_acc
            best_name = name
            best_pipeline = pipe

    root = get_root()
    model_dir = os.path.join(root, "src", "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(best_pipeline, model_path)

    print(f"Best model: {best_name}, val_accuracy={best_score:.4f}")
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()
