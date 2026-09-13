"""
train_model.py

Trains two candidate models on student_data.csv:
  - Linear Regression  (assumes a roughly straight-line relationship)
  - Random Forest      (can capture non-linear patterns and interactions)

Both are evaluated on a held-out test set they never trained on, and
whichever generalizes better is saved as student_model.pkl for the
FastAPI backend to load at prediction time.

Run this from anywhere:
    python train_model.py
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file, so it works no matter where you
# run the command from.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
DATA_PATH = THIS_DIR.parent / "data" / "student_data.csv"
MODEL_PATH = THIS_DIR / "student_model.pkl"

TARGET = "Exam_Score"
NUMERIC_FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity",
]
CATEGORICAL_FEATURES = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Turns raw columns into numbers a model can use.

    Numeric columns: fill any missing values with the column's median.
    Categorical columns (e.g. "High"/"Medium"/"Low"): fill missing values
    with the most common category, then one-hot encode — each category
    becomes its own 0/1 column, since models only understand numbers.

    A fresh instance is built for each model below so the two pipelines
    never share fitted state.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    # -----------------------------------------------------------------
    # 1. Load the data
    # -----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # -----------------------------------------------------------------
    # 2. Train / test split
    #    20% is held out completely. Neither model ever trains on it,
    #    so it measures real generalization to new students, not
    #    memorization of ones we've already seen.
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}\n")

    # -----------------------------------------------------------------
    # 3. Define the two candidates. Each is a full pipeline —
    #    preprocessing + model — so raw input (numbers and category
    #    strings) can go in one end and a prediction comes out the
    #    other, with no separate preprocessing step to keep in sync.
    # -----------------------------------------------------------------
    candidates = {
        "Linear Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]),
    }

    # -----------------------------------------------------------------
    # 4. Train and evaluate each candidate on the untouched test set
    # -----------------------------------------------------------------
    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        results[name] = r2
        print(name)
        print(f"  R^2  : {r2:.4f}   (share of score variation explained, 1.0 = perfect)")
        print(f"  MAE  : {mae:.2f} points  (average size of a miss)")
        print(f"  RMSE : {rmse:.2f} points  (like MAE, but penalizes big misses harder)")
        print()

    # -----------------------------------------------------------------
    # 5. Pick the winner by test-set R^2 — not training R^2, which
    #    would just reward whichever model memorized the data hardest.
    # -----------------------------------------------------------------
    winner_name = max(results, key=results.get)
    print(f"Winner: {winner_name}  (test R^2 = {results[winner_name]:.4f})")

    # -----------------------------------------------------------------
    # 6. Refit the winning model type on ALL the data (train + test
    #    combined). The test set already told us how well this model
    #    generalizes — now that the choice is made, there's no reason
    #    to withhold 20% of the data from the model that will actually
    #    serve predictions.
    # -----------------------------------------------------------------
    final_pipeline = candidates[winner_name]
    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
