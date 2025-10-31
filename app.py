from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap

# ===== Initialize FastAPI =====
app = FastAPI()

# ===== Load model =====
model = joblib.load("xgb_adasyn_model.pkl")

# ===== Initialize SHAP explainer =====
explainer = shap.TreeExplainer(model)

# ===== Input model structure =====
class ModelInput(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float        # Blood Sugar
    BodyTemp: float
    HeartRate: float

# ===== API Root =====
@app.get("/")
def root():
    return {"message": "Maternal Health Risk Prediction API is running"}

# ===== Prediction Endpoint =====
@app.post("/predict")
def predict(data: ModelInput):
    features = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                          data.BS, data.BodyTemp, data.HeartRate]])

    prediction = model.predict(features)[0]
    shap_values = explainer.shap_values(features)

    # Pair feature names with their SHAP importance values
    feature_names = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    feature_importance = dict(zip(feature_names, shap_values[0].tolist()))

    return {
        "prediction": int(prediction),
        "explanation": feature_importance
    }
