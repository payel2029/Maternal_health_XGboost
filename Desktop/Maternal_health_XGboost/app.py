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
    # Convert input to numpy array
    X = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                   data.BS, data.BodyTemp, data.HeartRate]])

    # Predict class
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]

    return {
        "risk_class": int(prediction),
        "probabilities": prediction_proba.tolist()
    }
