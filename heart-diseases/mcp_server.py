import mlflow.pyfunc
import pandas as pd
import joblib
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import os

import time

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("heart_disease_inference_monitoring_mcp")

mcp = FastMCP("Heart Disease Prediction API")


model = joblib.load("exports/model.pkl")

scaler = joblib.load("exports/min_scaler.pkl")

COLUMNS = [
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

@mcp.tool()
def predict_diabetes(
    age: int,
    sex: int,
    chest_pain_type: int,
    resting_blood_pressure: int,
    cholestrol: int,
    fasting_blood_sugar: int,
    resting_ecg: int,
    max_heart_rate: int,
    exercise_induced_angina: int,
    depression_induced_by_exercise: float,
    slope: int,
    major_vessels: int,
    thalasemia: int,
) -> dict:
    
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            start_time = time.time()
            # Convert to DataFrame (IMPORTANT: preserve column order)
            df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": chest_pain_type,
            "trestbps": resting_blood_pressure,
            "chol": cholestrol,
            "fbs": fasting_blood_sugar,
            "restecg": resting_ecg,
            "thalach": max_heart_rate,
            "exang": exercise_induced_angina,
            "oldpeak": depression_induced_by_exercise,
            "slope": slope,
            "ca": major_vessels,
            "thal": thalasemia
            }])
            
            df = df[COLUMNS]

            # Ensure column order matches training
            
            # Apply scaling
            scaled = scaler.transform(df)

            # Convert back to DataFrame (optional but clean)
            scaled_df = pd.DataFrame(scaled, columns=COLUMNS)

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
                "prediction": prediction,
                "interpretation": "High risk" if prediction == 1 else "Low risk"
            }

    except Exception as e:
            return {
                "error": str(e),
            }
            
if __name__ == "__main__":
    mcp.run("streamable-http")