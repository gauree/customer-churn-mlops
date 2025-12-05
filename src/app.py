import os

from flask import Flask, request, jsonify, Response, render_template
import joblib
import pandas as pd


app = Flask(__name__)


def get_root() -> str:
    """Return project root directory, consistent with existing code structure."""
    return os.path.dirname(os.path.dirname(__file__))


_MODEL = None


def load_model():
    """Lazily load and cache the trained model pipeline."""
    global _MODEL
    if _MODEL is None:
        root = get_root()
        model_path = os.path.join(root, "src", "model", "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run src/model/train.py first to train and save the model."
            )
        _MODEL = joblib.load(model_path)
    return _MODEL


@app.route("/", methods=["GET"])
def index() -> Response:
    """Serve the landing page for customer churn prediction."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Predict churn for one or more customers.

    Expected JSON body format:
    {
        "data": [
            {"feature1": value, "feature2": value, ...},
            {"feature1": value, "feature2": value, ...}
        ]
    }
    """
    try:
        model = load_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    payload = request.get_json(silent=True)
    if not payload or "data" not in payload:
        return jsonify({"error": "JSON body must contain 'data' key"}), 400

    records = payload["data"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'data' must be a non-empty list of records"}), 400

    # Convert list of dicts to DataFrame
    try:
        df = pd.DataFrame(records)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to construct DataFrame from input: {exc}"}), 400

    # Run predictions
    try:
        preds = model.predict(df)
        response = {"predictions": [int(p) for p in preds]}

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(df)[:, 1].tolist()
            response["churn_probability"] = probas
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(response), 200

@app.route("/openapi.json", methods=["GET"])
def openapi_spec() -> Response:
    """Minimal OpenAPI specification for this service."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Customer Churn Prediction API",
            "version": "1.0.0",
            "description": "Simple API to predict customer churn using a trained ML model.",
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"}
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/predict": {
                "post": {
                    "summary": "Predict customer churn",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "data": {
                                            "type": "array",
                                            "items": {"type": "object"},
                                        }
                                    },
                                    "required": ["data"],
                                },
                                "example": {
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
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "predictions": {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            },
                                            "churn_probability": {
                                                "type": "array",
                                                "items": {"type": "number", "format": "float"},
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"},
                    },
                }
            },
        },
    }
    return jsonify(spec)


@app.route("/docs", methods=["GET"])
def docs() -> Response:
    """Serve a simple Swagger UI page to explore the API."""
    html = """<!DOCTYPE html>
<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\" />\n  <title>Customer Churn API Docs</title>\n  <link rel=\"stylesheet\" type=\"text/css\" href=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css\" />\n</head>\n<body>\n  <div id=\"swagger-ui\"></div>\n  <script src=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js\"></script>\n  <script>\n    window.onload = () => {\n      window.ui = SwaggerUIBundle({\n        url: '/openapi.json',\n        dom_id: '#swagger-ui',\n      });\n    };\n  </script>\n</body>\n</html>"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
