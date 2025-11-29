from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np

class Sample(BaseModel):
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    O3: float
    temperature: float
    humidity: float
    wind_speed: float

app = FastAPI()
model = joblib.load("model/aqi_model.pkl")

@app.post("/predict")
def predict(sample: Sample):
    row = pd.DataFrame([{
        "PM2.5": sample.PM2_5,
        "PM10": sample.PM10,
        "NO2": sample.NO2,
        "SO2": sample.SO2,
        "CO": sample.CO,
        "O3": sample.O3,
        "temperature": sample.temperature,
        "humidity": sample.humidity,
        "wind_speed": sample.wind_speed
    }])
    pred = float(model.predict(row)[0])
    return {"AQI": pred}
