
#FasAPI Deployment

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('medical_cost_prediction_dataset.pkl')
encoder = joblib.load('encoder.pkl')

class InputData(BaseModel):
    age: int
    gender: str
    bmi: float
    smoker: str
    diabetes: int
    hypertension: int
    heart_disease: int
    asthma: int
    physical_activity_level: str
    daily_steps: int
    sleep_hours: float
    stress_level: int
    doctor_visits_per_year: int
    hospital_admissions: int
    medication_count: int
    insurance_type: str
    insurance_coverage_pct: int
    city_type: str
    previous_year_cost: float

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    # Encode categorical columns
    cat_cols = ['gender', 'smoker', 'physical_activity_level', 'insurance_type', 'city_type']
    df[cat_cols] = encoder.transform(df[cat_cols])
    prediction = model.predict(df)
    return {"predicted_annual_medical_cost": prediction[0]}

