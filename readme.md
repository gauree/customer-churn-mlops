
# Telco Customer Churn - MLOps project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-009688.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-0194E2.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.38.1-945DD6.svg)](https://dvc.org)

A complete and production-ready MLOps pipeline for predicting customer churn using machine learning.

📘 README.md — Telco Customer Churn Prediction (End-to-End MLOps Project)

A complete and production-ready MLOps pipeline for predicting customer churn using machine learning.

# 📑 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing & DVC Tracking](#data-preprocessing--dvc-tracking)
- [Model Development with MLflow Tracking](#model-development-with-mlflow-tracking)
- [Prefect Pipeline Orchestration](#prefect-pipeline-orchestration)
- [Repository Structure & Version Control](#repository-structure--version-control)
- [CI/CD using GitHub Actions](#cicd-using-github-actions)
- [Local Model Deployment (FastAPI/Streamlit)](#local-model-deployment-fastapistreamlit)
- [Containerization using Docker](#containerization-using-docker)
- [Local Monitoring using Evidently](#local-monitoring-using-evidently)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [MLOps Pipeline](#mlops-pipeline)
- [Contributing](#contributing)
- [License](#license)

# 🎯 **Project Overview**

## 🔍 **Objective**

The goal of this project is to **predict customer churn** for a telecom company based on:

- Historical billing data
- Service usage patterns
- Demographic information
- Subscription details

---

## 🧩 **Problem Definition**

The **Telco Customer Churn dataset** contains information about:

- Customer demographics
- Account information
- Subscribed services
- Monthly & total charges

### **Business Problem**
Telecom companies lose millions every year due to customer churn.  
Identifying customers likely to leave helps implement **proactive retention strategies**.

### **Machine Learning Task: Binary Classification**

- **Churn = Yes** → Customer is likely to leave
- **Churn = No** → Customer will stay

---

# 🧪 **Dataset**
## ** Why This Dataset?**
- ✔️ Benchmark dataset widely used for churn prediction
- ✔️ Rich and balanced set of real-world telecom features
- ✔️ Suitable for end-to-end **MLOps pipelines**
- ✔️ Allows meaningful:
    - Feature engineering
    - Hyperparameter tuning
    - MLflow experiment tracking
- ✔️ Clean, well-structured, and well-documented dataset

# 📊 **Exploratory Data Analysis (EDA)**

**`eda_app.py`** performs interactive EDA, including:

---

## ✔ **Missing Value Analysis**

- Identifies null or missing fields across all features
- Helps understand data quality before modeling

---

## ✔ **Distribution of Key Features**

### **Tenure**
### **Monthly Charges**
### **Contract Type**

Visualizing these helps reveal customer behavior patterns and spending trends.

---

## ✔ **Categorical Feature Breakdown**

- Payment methods
- Internet service types
- Dependents and partner status

These breakdowns provide insight into how customer groups differ across services.

---

## ✔ **Churn Ratio Visualization**

- Pie charts showing churn vs. non-churn proportions
- Bar charts grouped by demographics and services

This helps identify churn-heavy customer segments.

---

## ✔ **Correlation Heatmap**

Used to identify relationships between numerical variables and churn.

---

## 🧠 **Key Insights from EDA**

- Customers on **month-to-month contracts** tend to churn more
- **Electronic check** users have a significantly higher churn rate
- **High monthly charges** strongly correlate with increased churn

---

# 🧼 **Data Preprocessing & DVC Tracking**

Preprocessing script **`preprocess.py`** includes the following steps:

---

## ✔ **Dropping Duplicates**
Removes repeated rows to ensure clean and consistent training data.

---

## ✔ **Converting Numerical Columns**
Applies appropriate type casting for columns such as  
`tenure`, `MonthlyCharges`, and `TotalCharges`.

---

## ✔ **Encoding Categorical Features**
Uses:
- **OneHotEncoder** for multi-class categorical variables
- **LabelEncoder** for binary categorical variables

This ensures that machine learning models can process categorical data effectively.

---

## ✔ **Train–Test Split**
Splits the dataset into training and testing subsets to enable proper model evaluation.

---

## ✔ **Saving Processed Data**
Processed outputs are saved under:

```
data/processed/
```

---

# 📦 **DVC (Data Version Control)**

DVC is used to track:

- Raw dataset
- Processed dataset
- Feature engineering artifacts

This enables reproducibility across the ML pipeline.

---

## **Commands Used**

```bash
dvc add data/raw/Telco-Customer-Churn.csv
git add data/raw/Telco-Customer
git commit -m "Track raw Telco dataset"
```
---
```bash
# Pipeline definition stored in:
dvc.yaml
.dvc/config
```
Now the pipeline is reproducible anywhere.

---

# 🤖 **Model Development with MLflow Tracking**

Training script **`train.py`** includes the following machine learning models:

---

## ✔ **Logistic Regression**

## ✔ **Random Forest**

## ✔ **XGBoost / Gradient Boosting**

## ✔ **Support Vector Machine (Optional)**

---

# 📊 **MLflow Logging**

MLflow tracks and stores:

- **Parameters**
- **Metrics** (Accuracy, F1-Score, ROC-AUC)
- **Confusion Matrix**
- **Model Artifacts**
- **Best Model in Registry**

These logs allow reproducible experiments, model comparison, and lifecycle management.

---

## 🚀 **Start MLflow UI**

```bash
mlflow ui --port 5000
```

---

# 🔄 **Prefect Pipeline Orchestration**

Workflow pipeline is composed of multiple modular **Prefect tasks**, including:

---

## ✔ **Pipeline Tasks**

- `load_raw_data()`
- `run_eda()`
- `data_preprocessing()`
- `train_model()`
- `evaluate_model()`
- `register_model()`

---

## 🧩 **Full Prefect Flow Definition**

```python
@flow(name="churn-mlops-pipeline")
def pipeline():
    raw = load_raw_data()
    eda_results = run_eda(raw)
    processed = data_preprocessing(raw)
    model = train_model(processed)
    evaluate_model(model)
```

---

## ▶️ **Run Prefect Orchestration**

```bash
prefect orion start
```

---

# 🧱 **Repository Structure & Version Control**

Following standard **MLOps best practices**, repository structure is organized as:

```
Telco-Churn/
│── src/
│   ├── data/
│   ├── eda/
│   ├── models/
│   ├── pipelines/
│   ├── api/
│── data/
│   ├── raw/
│   ├── processed/
│── models/
│── mlruns/
│── dvc.yaml
│── requirements.txt
│── Dockerfile
│── README.md
```
---

## 🌱 **Git Branching Strategy**

- **main** → Production-ready code
- **dev** → Active development
- **feature/*** → New modules and experimental features

---
# ⚙️ **CI/CD using GitHub Actions**

CI workflow automates the complete MLOps lifecycle, including:

---

## 🚀 **CI Workflow Includes**

- Install dependencies
- Run unit tests
- Run linting
- Execute DVC pipeline
- Train & evaluate the model
- Push the best model to MLflow registry

---

/ .github/workflows/mlops-ci.yaml`) includes:

```yaml
- name: Run Tests
  run: pytest -v

- name: Run DVC Pipeline
  run: dvc repro
```
---

# 🚀 **Local Model Deployment (FastAPI / Streamlit)**

---

## ⚡ **FastAPI Prediction Server**

### **Start the API Server**

```bash
uvicorn src.api.app:app --reload
```

### **Available Endpoints**

- `/predict`
- `/health`
- `/batch_predict`

### **Interactive API Documentation**

Visit:

```
http://localhost:8000/docs
```

---

## 📊 **Streamlit Dashboard**

Run the Streamlit app:

```bash
streamlit run app.py
```
---
# 🐳 **Containerization using Docker**

---

## 🛠️ **Build Docker Image**

```bash
docker build -t telco-churn-api .
```

---

## ▶️ **Run Docker Container**

```bash
docker run -p 8000:8000 telco-churn-api
```

---

📦 **Run with DVC + MLflow Volumes (Docker Compose)**

```bash
docker-compose up --build
```
---
# 📈 **Local Monitoring using Evidently**

Automated monitoring pipeline includes **data drift detection** and an interactive dashboard.

---

## ▶️ **Run Monitoring Script**

```bash
python monitoring/evidently_report.py
```

---

## 📊 **Outputs Generated**

- **Data Drift Report**
- **Target Drift Analysis**
- **PSI (Population Stability Index) Values**
- **Interactive HTML Dashboard**

---

## ⚙️ **Installation**

```bash
pip install -r requirements.txt
dvc pull
```

---

# ▶️ **Usage**

---

## 📊 **Run EDA**

```bash
python eda_app.py
```

---

## 🧹 **Preprocess Data**

```bash
python preprocess.py
```

---

## 🤖 **Train Model**

```bash
python train.py
```

---

## 🔮 **Predict via API**

```bash
curl -X POST http://localhost:8000/predict -d '{...}'
```

---

# 📘 **API Documentation**

---

## 📨 **POST /predict**

### **Request Body**

```json
{
  "gender": "Female",
  "Partner": "Yes",
  "tenure": 5,
  "InternetService": "DSL"
  ...
}
```

---

### **Response**

```json
{
  "churn_probability": 0.81,
  "prediction": "Yes"
}
```

---
# 🧪 **Model Performance**

### **Metrics Evaluated**

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Confusion Matrix** is included in the MLflow artifacts.

---

# 🧪 **Testing**

Run tests using:

```bash
pytest -v
```

---

# 🐋 **Docker Deployment**

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

# 📁 **Project Structure**

```
src/
├── eda_app.py
├── preprocess.py
├── train.py
├── api/
│   └── app.py
data/
├── raw/
├── processed/
models/
mlruns/
dvc.yaml
Dockerfile
README.md
```

---

# 🔄 **MLOps Pipeline**

```
Raw Data
 → EDA
 → Preprocessing
 → Model Training (MLflow)
 → Evaluation
 → Deployment
 → Monitoring (Evidently)
```

---

# 🙌 **Contributing**

- Fork the repository
- Create a feature branch
- Submit a pull request

---

# 📜 **License**

**MIT License**

---


