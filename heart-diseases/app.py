import mlflow.pyfunc
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os

import time

import mlflow
import os

def wait_for_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI")

    for i in range(10):
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.get_experiment_by_name("test")
            print("Connected to MLflow")
            return
        except Exception as e:
            print(f"Waiting for MLflow... ({i+1}/10)")
            time.sleep(3)

    raise Exception("MLflow not available")

app = FastAPI(title="Heart Disease Prediction API")


@app.on_event("startup")
def startup():
    wait_for_mlflow()

    

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("heart_disease_inference_monitoring")



scaler = joblib.load("exports/min_scaler.pkl")

# -------------------------
# Define input schema
# -------------------------
class HeartDiseaseInput(BaseModel):
    age: int = Field(..., ge=0)
    sex: int
    chest_pain_type: int
    resting_blood_pressure: int
    cholestrol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_induced_angina: int
    depression_induced_by_exercise: float
    slope: int
    major_vessels: int
    thalasemia: int
# -------------------------
# App
# -------------------------



model = joblib.load("exports/model.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(data: HeartDiseaseInput):
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            start_time = time.time()
            # Convert to DataFrame (IMPORTANT: preserve column order)
            df = pd.DataFrame([{
            "age": data.age,
            "sex": data.sex,
            "cp": data.chest_pain_type,
            "trestbps": data.resting_blood_pressure,
            "chol": data.cholestrol,
            "fbs": data.fasting_blood_sugar,
            "restecg": data.resting_ecg,
            "thalach": data.max_heart_rate,
            "exang": data.exercise_induced_angina,
            "oldpeak": data.depression_induced_by_exercise,
            "slope": data.slope,
            "ca": data.major_vessels,
            "thal": data.thalasemia
            }])

            # Ensure column order matches training
            columns = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
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