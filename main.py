
# Step 13: Student Performance Prediction API

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict student exam scores using a trained ML model.",
    version="1.0.0"
)


# Define the input data required for prediction
class StudentData(BaseModel):
    Hours_Studied: int = Field(ge=0, le=24)
    Attendance: int = Field(ge=0, le=100)
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: int = Field(ge=0, le=24)
    Previous_Scores: int = Field(ge=0, le=100)
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int = Field(ge=0)
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Physical_Activity: int = Field(ge=0)
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str


# Basic health-check endpoint
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: StudentData):
    student = pd.DataFrame([data.model_dump()])
    prediction = model.predict(student)[0]

    return {
        "predicted_exam_score": round(float(prediction), 2)
    }
