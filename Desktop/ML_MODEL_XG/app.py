# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Load the XGBoost model and define class mapping
# ------------------------
try:
    model = joblib.load("xgb_adasyn_model.pkl")
    logger.info("Model loaded successfully")
    
    # Define the class mapping based on typical label encoding
    # Since we don't have the original classes_, we'll use the most common mapping
    class_mapping = {
        0: "low risk",
        1: "mid risk", 
        2: "high risk"
    }
    
    logger.info(f"Using class mapping: {class_mapping}")
        
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

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
    
    # Map to string label
    risk_label = class_mapping.get(prediction, "unknown risk")

    return {
        "risk_class": int(prediction),
        "risk_label": risk_label,  # This is the important addition!
        "probabilities": prediction_proba.tolist(),
        "class_mapping": class_mapping  # Send mapping to frontend for verification
    }