import mlflow.pyfunc
import pandas as pd
import joblib
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import os

import time

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("diabetes_inference_monitoring_mcp")

mcp = FastMCP(title="Diabetes Prediction API")

# -------------------------
# Load model + scaler
# -------------------------
MODEL_URI = "models:/diabetes_rf_model/Latest"
model = mlflow.pyfunc.load_model(MODEL_URI)

scaler = joblib.load("exports/min_scaler.pkl")

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

@mcp.tool()
def predict_diabetes(
    Pregnancies: int,
    Glucose: float,
    BloodPressure: float,
    SkinThickness: float,
    Insulin: float,
    BMI: float,
    DiabetesPedigreeFunction: float,
    Age: int,
) -> dict:
    
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            start_time = time.time()
            # Convert to DataFrame (IMPORTANT: preserve column order)
            df = pd.DataFrame([{
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age,
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
    mcp.run("streamable-http", host="0.0.0.0", port=8001)