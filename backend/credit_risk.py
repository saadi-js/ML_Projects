from pathlib import Path

import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from fastapi.staticfiles import StaticFiles




project_path = Path(__file__).resolve().parent.parent
model_path = project_path / "model" / "credit_risk" / "credit_model.pkl"
threshold_path = project_path / "model" / "credit_risk" / "threshold.pkl"

credit_model = joblib.load(model_path)
credit_threshold = joblib.load(threshold_path)


ml_model={}
@asynccontextmanager
async def lifespan(app:FastAPI):
    ml_model['credit_model']=credit_model
    ml_model['credit_threshold']=credit_threshold
    yield
    ml_model.clear()



app=FastAPI(lifespan=lifespan)

class LoanApplication(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int



@app.post("/predict")
async def predict_loan_status(application: LoanApplication):
    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([application.dict()])

    prediction_proba = ml_model['credit_model'].predict_proba(input_data)[:, 1][0]
    prediction = int(prediction_proba > ml_model['credit_threshold'])

    return {
        "default_probabilty":float(prediction_proba),
        "prediction": prediction,
        "threshold": float(ml_model['credit_threshold']),
        "Result": "High Risk" if prediction == 1 else "Low Risk"
        }


frontend_path = project_path / "frontend" / "credit_risk"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="credit-risk-frontend")