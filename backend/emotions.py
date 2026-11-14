from keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from keras.models import load_model
import pickle
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import HTTPException
import numpy as np

import re

project_path = Path(__file__).resolve().parent.parent
model_path = project_path / "model" / "emotion_model" / "BiGRU_Model.keras"
tokenizer_path = project_path / "model" / "emotion_model" / "tokenizer.pkl"

max_seq_length = 50

Emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

EMOTION_EMOJI={
    'anger': '😠',
    'fear': '😨',
    'joy': '😄',
    'love': '❤️',
    'sadness': '😢',
    'surprise': '😲'
}

def process_text(text:str)-> str:
    """
    Preprocess the input text for emotion prediction.
    """

    text = text.lower()
    text=re.sub("'","",text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


class InputText(BaseModel):
    text: str=Field(
        ...,
        min_length=1, 
        max_length=2000, 
        description="Input text for emotion prediction",
        json_schema_extra={"example": "I am feeling very happy today!"})



class OutputEmotion(BaseModel):
    text:str
    emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class healthResponse(BaseModel):
    status:str
    model_loaded:bool




"""
Model Loading and Lifespan Management
"""
dl_model={}
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    dl_model["bigru"] = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
       dl_model["tokenizer"]=pickle.load(file)
    print("Model and tokenizer loaded.")
    yield 
    dl_model.clear()



app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "Text_model"

#API EndPoint
@app.get('/',include_in_schema=False)
def serve_ui():
    return FileResponse(frontend_path / "index.html")


@app.get("/health", response_model=healthResponse)
def health_check():
    return healthResponse(status="healthy", model_loaded=bool(dl_model))


@app.post("/predict", response_model=OutputEmotion)
def predict_emotion(input_data: InputText):
    BiGRU_model = dl_model.get("bigru")
    tokenizer = dl_model.get("tokenizer")
    if BiGRU_model is None or tokenizer is None:
       raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

    # Preprocess the input text
    processed_text = process_text(input_data.text)
    tokenized_text = tokenizer.texts_to_sequences([processed_text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_seq_length, padding='post', truncating='post')
    probabilities = BiGRU_model.predict(padded_text)[0]
    predicted_index = probabilities.argmax()
    predicted_emotion = Emotion_labels[predicted_index]
    confidence = probabilities[predicted_index]
    all_probabilities={label: prob for label, prob in zip(Emotion_labels, probabilities)}

    return OutputEmotion(
        text=input_data.text,
        emotion=predicted_emotion,
        confidence=confidence,
        all_probabilities=all_probabilities
    )


app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
