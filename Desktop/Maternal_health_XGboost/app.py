from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap

app = FastAPI()

# Load the trained XGBoost model
model = joblib.load("xgb_adasyn_model.pkl")

class ModelInput(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float        # Blood Sugar
    BodyTemp: float
    HeartRate: float

@app.get("/")
def root():
    return {"message": "Maternal Health Risk Prediction API is running"}

@app.post("/predict")
def predict(data: ModelInput):
    # Prepare input
    features = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                          data.BS, data.BodyTemp, data.HeartRate]])

    # Make prediction
    prediction = int(model.predict(features)[0])

    # Try generating explanation safely
    explanation = {}
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        feature_names = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
        # Handle multiclass shap output
        if isinstance(shap_values, list):
            shap_values = shap_values[prediction]
        explanation = dict(zip(feature_names, shap_values[0].tolist()))
    except Exception as e:
        explanation = {"error": str(e)}

    return {
        "prediction": prediction,
        "explanation": explanation
    }
