import mlflow.pyfunc
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os

import time

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("diabetes_inference_monitoring")

# -------------------------
# Load model + scaler
# -------------------------
MODEL_URI = "models:/diabetes_rf_model/Latest"
model = mlflow.pyfunc.load_model(MODEL_URI)

scaler = joblib.load("exports/min_scaler.pkl")

# -------------------------
# Define input schema
# -------------------------
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int = Field(..., ge=0)

# -------------------------
# App
# -------------------------
app = FastAPI(title="Diabetes Prediction API")


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(data: DiabetesInput):
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            start_time = time.time()
            # Convert to DataFrame (IMPORTANT: preserve column order)
            df = pd.DataFrame([data.dict()])

            # Ensure column order matches training
            columns = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
            df = df[columns]

            # Apply scaling
            scaled = scaler.transform(df)

            # Convert back to DataFrame (optional but clean)
            scaled_df = pd.DataFrame(scaled, columns=columns)

            # Predict
            prediction = model.predict(scaled_df)[0]
            
            latency = time.time() - start_time
            
            mlflow.log_metric("latency", latency)
            mlflow.log_param("model_version", "latest")

            # log inputs (optional but powerful)
            for col in df.columns:
                mlflow.log_param(f"input_{col}", df[col].iloc[0])

            mlflow.log_metric("prediction", int(prediction))

            return {
                "prediction": int(prediction),
                "status": "success"
            }

    except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }