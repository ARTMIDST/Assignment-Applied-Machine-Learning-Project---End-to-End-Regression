from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("medical_cost_model.pkl")
encoder = joblib.load("encoder.pkl")

FEATURE_ORDER = [
    'age', 'gender', 'bmi', 'smoker', 'diabetes', 'hypertension',
    'heart_disease', 'asthma', 'physical_activity_level', 'daily_steps',
    'sleep_hours', 'stress_level', 'doctor_visits_per_year',
    'hospital_admissions', 'medication_count', 'insurance_type',
    'insurance_coverage_pct', 'city_type', 'previous_year_cost'
]

CAT_COLS = [
    'gender', 'smoker', 'physical_activity_level',
    'insurance_type', 'city_type'
]

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

    df[CAT_COLS] = encoder.transform(df[CAT_COLS])

    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].median())

    df = df[FEATURE_ORDER]

    prediction = model.predict(df)

    return {"predicted_annual_medical_cost": float(prediction[0])}
