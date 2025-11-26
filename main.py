from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class Input(BaseModel):
    value: float

@app.post("/predict")
def predict(data: Input):
    pred = model.predict([[data.value]])[0]
    return {"prediction": pred}
