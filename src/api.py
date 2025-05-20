import os
import joblib
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.testclient import TestClient  # For tests


# Load your offline pickle
MODEL_PATH = os.path.join("models", "xgb_best.pkl")
model = joblib.load(MODEL_PATH)

# Structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

# Input schema
class Customer(BaseModel):
    CreditScore:      int
    Geography:        str
    Gender:           str
    Age:              int
    Tenure:           int
    Balance:          float
    NumOfProducts:    int
    HasCrCard:        int
    IsActiveMember:   int
    EstimatedSalary:  float

app = FastAPI(
    title="Bank Churn Prediction API",
    version="1.0.0",
)

@app.get("/", tags=["home"])
def home():
    return {"message": "Bank Churn Prediction API is up"}

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

@app.post("/predict", tags=["predict"])
def predict(c: Customer):
    # Convert to DataFrame
    df = pd.DataFrame([c.dict()])

    # One-hot encode
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Ensure all dummy columns exist
    for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
        if col not in df.columns:
            df[col] = 0

    # Reorder features
    ordered_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary",
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ]
    X = df[ordered_cols].values

    # Log & predict
    logger.info(f"Input → {X.tolist()}")
    try:
        proba = model.predict_proba(X)[0, 1]
        logger.info(f"Output → {proba}")
        return {"churn_probability": float(proba)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TESTS

client = TestClient(app)

def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert "API is up" in r.json().get("message", "")

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

# A sample payload , use ASSERTS
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
    """POST /predict returns a valid probability for a good payload"""
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0

def test_predict_missing_field():
    """Missing a required field yields a 422"""
    bad = sample.copy()
    bad.pop("Age")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_predict_type_error():
    """Wrong type for a numeric field yields a 422"""
    bad = sample.copy()
    bad["Balance"] = "not_a_number"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_predict_method_not_allowed():
    """GET /predict should be Method Not Allowed (405)"""
    r = client.get("/predict")
    assert r.status_code == 405

def test_unknown_endpoint_returns_404():
    """Requests to nonexistent paths return 404"""
    r = client.get("/not-an-endpoint")
    assert r.status_code == 404