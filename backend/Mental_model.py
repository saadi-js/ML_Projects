from pathlib import Path
from typing_extensions import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware




model_path = Path(
    r"C:\Users\user\OneDrive\Desktop\MentalModel\model\Mental_health_Model.pkl"
)
model = joblib.load(model_path)



app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def hello():
    return {"message": "hello"}


class InputData(BaseModel):
    Age : int = Field(..., ge=10, le=100, description="Age must be between 10 and 100")
    Gender: str = Literal['Male', 'Female', 'Other']
    Country: str
    Academic_Level:str=Literal['High School', 'Undergraduate', 'Graduate']
    Most_Used_Platform:str=Literal['Facebook','LinkedIn','Instagram','Snapchat','Twitter','YouTube','TikTok','LINE','KakaoTalk','VKontakte','WhatsApp','WeChat']
    Purpose_Of_Use:str= Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours:float=Field(..., ge=0, le=24)
    Daily_Unlocks:int=Field(..., ge=0)
    Study_Hours:float=Field(..., ge=0, le=24)
    Physical_Activity_Hours:float=Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night:float=Field(..., ge=0, le=24)
    Stress_Level:str=Literal['Low', 'Medium', 'High', 'Very High']

# Describe What we send back



top_country =  ['Other',
    'India',
    'USA',
    'Canada',
    'Australia',
    'UK',
    'Germany',
    'Mexico',
    'Turkey',
    'France']

class PredictionResponse(BaseModel):
    predected_mental_health_score:float


@app.post('/predict',response_model=PredictionResponse)
def predict(data:InputData):
    country_group= data.Country if data.Country in top_country else "Other"

    input_row = pd.DataFrame([{
        'Age':data.Age,
        'Gender':data.Gender,
        'Country':data.Country,
        'Academic_Level':data.Academic_Level,
        'Most_Used_Platform':data.Most_Used_Platform,
        'Purpose_Of_Use':data.Purpose_Of_Use,
        'Avg_Daily_Usage_Hours':data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks':data.Daily_Unlocks,
        'Study_Hours':data.Study_Hours,
        'Physical_Activity_Hours':data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night':data.Sleep_Hours_Per_Night,
        'Stress_Level':data.Stress_Level,
        'Grouped_Country':country_group
    }])

    prediction=model.predict(input_row)[0]
    return PredictionResponse(predected_mental_health_score=round(float(prediction),2))


    

    
