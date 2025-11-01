# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ------------------------
# Define input schema
# ------------------------
class PatientData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float        # Blood Sugar
    BodyTemp: float
    HeartRate: float

# ------------------------
# Initialize FastAPI
# ------------------------
app = FastAPI(title="Maternal Health Risk Prediction API")

# ------------------------
# Load the XGBoost model
# ------------------------
model = joblib.load("xgb_adasyn_model.pkl")

# ------------------------
# Home route
# ------------------------
@app.get("/")
def home():
    return {"message": "Maternal Health Risk Prediction API is running"}

# ------------------------
# Prediction route
# ------------------------
@app.post("/predict")
def predict_risk(data: PatientData):
    import pandas as pd
    import shap

    # Convert input to DataFrame for model and SHAP
    X = pd.DataFrame([{
        "Age": data.Age,
        "SystolicBP": data.SystolicBP,
        "DiastolicBP": data.DiastolicBP,
        "BS": data.BS,
        "BodyTemp": data.BodyTemp,
        "HeartRate": data.HeartRate
    }])

    # Predict class (numerical label)
    prediction = int(model.predict(X)[0])
    prediction_proba = model.predict_proba(X)[0].tolist()

    # Map numeric class → text label
    risk_labels = {0: "low risk", 1: "mid risk", 2: "high risk"}
    risk_level = risk_labels.get(prediction, "unknown")

    # Try SHAP explanation
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        explanation = dict(zip(X.columns, shap_values.values[0].tolist()))
    except Exception as e:
        explanation = {"note": "SHAP explanation unavailable", "error": str(e)}

    return {
        "prediction": risk_level,
        "probabilities": {
            "low": round(prediction_proba[0], 3),
            "mid": round(prediction_proba[1], 3),
            "high": round(prediction_proba[2], 3)
        },
        "explanation": explanation
    }
