"""
schemas.py

Pydantic models describing what the API accepts (StudentInput) and what
it returns (PredictionOutput). FastAPI uses these to validate every
request automatically — if a field is missing, the wrong type, or not
one of the allowed categories, the caller gets a clear 422 error before
any of main.py's own code runs.

Field names match the training data's column names exactly (e.g.
Hours_Studied, not hours_studied) on purpose: the trained pipeline in
student_model.pkl expects a DataFrame with these exact column names, so
keeping the names identical end-to-end avoids a renaming step that could
silently drift out of sync with the model.
"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class StudentInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Hours_Studied": 20,
                "Attendance": 85,
                "Previous_Scores": 75,
                "Tutoring_Sessions": 1,
                "Sleep_Hours": 7,
                "Physical_Activity": 3,
                "Extracurricular_Activities": "Yes",
                "Motivation_Level": "Medium",
                "Parental_Involvement": "Medium",
                "Parental_Education_Level": "College",
                "Family_Income": "Medium",
                "Access_to_Resources": "Medium",
                "Internet_Access": "Yes",
                "Teacher_Quality": "Medium",
                "School_Type": "Public",
                "Distance_from_Home": "Near",
                "Peer_Influence": "Positive",
                "Learning_Disabilities": "No",
                "Gender": "Male",
            }
        }
    )

    # --- Study behavior ---
    Hours_Studied: float = Field(..., ge=0, le=168, description="Hours studied per week")
    Attendance: float = Field(..., ge=0, le=100, description="Attendance percentage")
    Previous_Scores: float = Field(..., ge=0, le=100, description="Score on a previous exam")
    Tutoring_Sessions: float = Field(..., ge=0, le=50, description="Tutoring sessions attended")

    # --- Lifestyle ---
    Sleep_Hours: float = Field(..., ge=0, le=24, description="Average hours of sleep per day")
    Physical_Activity: float = Field(..., ge=0, le=40, description="Hours of physical activity per week")
    Extracurricular_Activities: Literal["Yes", "No"]
    Motivation_Level: Literal["Low", "Medium", "High"]

    # --- Family background ---
    Parental_Involvement: Literal["Low", "Medium", "High"]
    Parental_Education_Level: Literal["High School", "College", "Postgraduate"]
    Family_Income: Literal["Low", "Medium", "High"]

    # --- School / learning environment ---
    Access_to_Resources: Literal["Low", "Medium", "High"]
    Internet_Access: Literal["Yes", "No"]
    Teacher_Quality: Literal["Low", "Medium", "High"]
    School_Type: Literal["Public", "Private"]
    Distance_from_Home: Literal["Near", "Moderate", "Far"]

    # --- Other ---
    Peer_Influence: Literal["Negative", "Neutral", "Positive"]
    Learning_Disabilities: Literal["Yes", "No"]
    Gender: Literal["Male", "Female"]


class PredictionOutput(BaseModel):
    predicted_exam_score: float = Field(..., description="Predicted exam score")
