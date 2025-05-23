# tests/test_api.py
import os
import joblib
import pandas as pd
import logging

from fastapi.testclient import TestClient
from src.api import app  # import the FastAPI app

client = TestClient(app)

def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert "up" in r.json().get("message", "").lower()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

sample = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
}

def test_predict_success():
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0

def test_predict_missing_field():
    bad = sample.copy(); bad.pop("Age")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_predict_type_error():
    bad = sample.copy(); bad["Balance"] = "oops"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_predict_method_not_allowed():
    r = client.get("/predict")
    assert r.status_code == 405

def test_unknown_endpoint():
    r = client.get("/not-an-endpoint")
    assert r.status_code == 404

