import os
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
# Load your offline pickle from models/xgb_best.pkl 
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
@app.post("/predict", tags=["predict"])
def predict(c: Customer):

    # Convert the input dict to a DataFrame
    df = pd.DataFrame([c.dict()])

    # One-hot encode exactly as in training
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Ensure all dummy columns exist (for unseen combinations)
    for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training order
    ordered_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary",
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ]
    X = df[ordered_cols].values

    logger.info(f"Input → {X.tolist()}")
    try:
        proba = model.predict_proba(X)[0, 1]
        logger.info(f"Output → {proba}")
        return {"churn_probability": float(proba)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
