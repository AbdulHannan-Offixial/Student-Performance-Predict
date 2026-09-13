"""
main.py

The FastAPI application. Loads the trained model pipeline once at
startup, exposes a single POST /predict endpoint, and allows the
Netlify frontend to call it over HTTP/JSON (CORS enabled below).

Run locally with:
    uvicorn main:app --reload
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import StudentInput, PredictionOutput

# ---------------------------------------------------------------------------
# Load the trained model once, when the server starts — not on every
# request, since reading from disk on every call would be needlessly slow.
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "student_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Student Performance Predictor",
    description="Predicts a student's exam score from study habits, lifestyle, and background factors.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS: without this, a browser blocks the Netlify frontend from calling
# this API, since they live on different domains. "*" is fine for
# development; once you know your Netlify URL, swap it in here for
# tighter security in production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with your Netlify URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Student Performance Predictor API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    """
    Takes one student's details and returns a predicted exam score.
    By the time this function runs, Pydantic has already checked every
    field's type and range — a request that reaches here is well-formed.
    """
    # The model was trained on a pandas DataFrame with these exact column
    # names, so we build a one-row DataFrame the same way before predicting.
    input_df = pd.DataFrame([student.model_dump()])

    try:
        raw_prediction = model.predict(input_df)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Linear regression can extrapolate past realistic bounds for extreme
    # inputs (e.g. very high hours studied). Clamp to a sane exam-score
    # range rather than returning something like -4 or 137.
    clamped_prediction = max(0.0, min(100.0, float(raw_prediction)))

    return PredictionOutput(predicted_exam_score=round(clamped_prediction, 2))
