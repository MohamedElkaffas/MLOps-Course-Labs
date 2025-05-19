# Bank Customer Churn Prediction - MLOps Lab

## 1 · Project Overview

This repo demonstrates a full MLOps cycle on the Bank Customer Churn Prediction dataset.Key goals:

Automate experiment tracking with MLflow.

Run & compare multiple models (Logistic Regression, Random Forest, XGBoost).

Register two models in the MLflow Model Registry – one in Staging as a lightweight baseline, and one in Production for deployment.

## 2 · Dataset

Kaggle: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/dataCSV placed at data/Churn_Modelling.csv.Target column: Exited (1 = customer left, 0 = retained).

## 3 · Environment Setup

conda create -n churn_prediction python=3.12 -y
conda activate churn_prediction
pip install -r requirements.txt  # or install manually
pip install mlflow scikit-learn pandas numpy xgboost

Note Git & Jupyter need to be on your PATH.

## 4 · Repo Structure

MLOps-Course-Labs/
├── data/                  # dataset CSV
├── screenshots/           # MLflow UI captures for grading
├── src/
│   └── train.py           # all training logic & MLflow logging
├── mlruns/                # local MLflow artefacts

## 5 · Running Experiments

# from repo root
python src/train.py              # logs 3 runs to MLflow
mlflow ui --port 5000            # open http://localhost:5000

train.py automatically:

Encodes categorical features (Geography, Gender)

Splits data 80 / 20

Logs runs named LR_baseline, RF_100_6, XGB_100_0.1

## 6 · Results & Metrics

Run Name

Accuracy

LR_baseline

0.801 5

RF_100_6

0.858 0

XGB_100_0.1

0.866 5

## 7 · Model Registry & Justification

Stage

Version

Run ID

Rationale

Staging

v1

LR_baseline

Logistic Regression is simple, interpretable & fast – a sanity‑check reference for further model iterations.

Production

v2

RF_100_6

Random Forest improved accuracy by +5.7 pp over baseline with modest complexity, beating LR while training quickly and avoiding XGBoost’s heavier dependencies.

XGBoost achieved the very best metric, but RF was selected for Production due to its smaller footprint and faster inference in our target environment.

Registration is handled programmatically via MlflowClient() at the end of src/train.py.

## 8 · Screenshots

All required UI captures are under screenshots/:

01_runs_table.png – run list

02_accuracy_chart.png – comparison chart

03_model_registry.png – Staging vs Production

## 9 · Reproducibility

Seeded splits (random_state=42).

Exact package versions in requirements.txt .

Every run stores its artefacts in MLflow (mlruns/).

## 10 · Next Steps

Add proper evaluation signatures & input examples when logging models.

Introduce a CI pipeline (GitHub Actions) that runs mlflow run on each push.

Containerise the Production model with mlflow models build-docker for deployment to Azure Container Apps.
