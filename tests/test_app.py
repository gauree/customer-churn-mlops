import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app as app_module  
flask_app = app_module.app


class DummyModel:
    def predict(self, X):
        n = len(X)
        return np.ones(n, dtype=int)  

    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.2, 0.8], (n, 1))


@pytest.fixture
def client():
    """Flask test client fixture."""
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


def test_predict_success_single_record(monkeypatch, client):
    monkeypatch.setattr(app_module, "load_model", lambda: DummyModel())

    payload = {
        "data": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        ]
    }

    resp = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] in (0, 1)

    assert "churn_probability" in body
    assert len(body["churn_probability"]) == 1


def test_predict_success_batch(monkeypatch, client):
    monkeypatch.setattr(app_module, "load_model", lambda: DummyModel())

    payload = {
        "data": [
            {"a": 1},
            {"a": 2},
            {"a": 3},
        ]
    }

    resp = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["predictions"]) == 3
    assert len(body["churn_probability"]) == 3


def test_predict_bad_payload_missing_data_key(client):
    resp = client.post(
        "/predict",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "error" in body


def test_predict_bad_payload_empty_list(monkeypatch, client):
    monkeypatch.setattr(app_module, "load_model", lambda: DummyModel())

    resp = client.post(
        "/predict",
        data=json.dumps({"data": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "error" in body


def test_predict_model_missing_returns_500(monkeypatch, client):
    def raise_not_found():
        raise FileNotFoundError("Model file not found")

    monkeypatch.setattr(app_module, "load_model", raise_not_found)

    payload = {"data": [{"a": 1}]}
    resp = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 500
    body = resp.get_json()
    assert "error" in body
