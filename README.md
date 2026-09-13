# Student Performance Predictor

A machine learning web app that estimates a student's exam score from their study habits, lifestyle, and background — and shows which factors actually move the needle, rather than relying on intuition.

## How it works

```
┌───────────────────────────┐
│  Frontend (HTML/JS)        │
│  hosted on Netlify         │
└─────────────┬───────────────┘
              │ HTTP / JSON
              ▼
┌───────────────────────────┐
│  FastAPI REST API          │
│  containerized with Docker │
└─────────────┬───────────────┘
              │
              ▼
┌───────────────────────────┐
│  scikit-learn model        │
│  (student_model.pkl)       │
└─────────────┬───────────────┘
              │ prediction
              ▼
         JSON response
```

The frontend collects 19 details about a student and sends them as JSON to a FastAPI backend. The backend runs them through a trained scikit-learn pipeline and returns a predicted exam score, which the frontend displays.

## The model

Two candidates were trained and compared on a held-out test set before either was trusted:

| Model | R² (test) | MAE | RMSE |
|---|---|---|---|
| **Linear Regression** (saved as `student_model.pkl`) | **0.770** | **0.45 pts** | **1.80 pts** |
| Random Forest | 0.670 | 1.08 pts | 2.16 pts |

Linear regression won: the real relationship between these features and exam score turned out to be close enough to linear that Random Forest's extra flexibility didn't help — it likely fit noise instead of signal. The full pipeline (missing-value imputation → one-hot encoding of categorical features → regression) is saved as a single `.pkl` file, so the API can hand it raw form values directly.

## Dataset

[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) — 6,607 students, 20 columns, released under **CC0: Public Domain**. It covers four groups of factors:

- **Study habits** — hours studied, attendance, previous scores, tutoring sessions
- **Lifestyle** — sleep, physical activity, extracurricular activities, motivation
- **Family background** — parental involvement, education level, income
- **School environment** — access to resources, internet access, teacher quality, school type, distance from home, peer influence, learning disabilities, gender

Target variable: `Exam_Score`.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript |
| Backend | FastAPI, Pydantic |
| Machine learning | scikit-learn, pandas, joblib |
| Deployment | Docker, Docker Compose |

## Project structure

```
student-performance-predictor/
├── backend/
│   ├── main.py              # FastAPI app and /predict endpoint
│   ├── schemas.py           # Pydantic request/response models
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── model/
│   │   ├── train_model.py   # Trains, compares, and saves the model
│   │   └── student_model.pkl
│   └── data/
│       └── student_data.csv
├── docker-compose.yml
├── frontend/
│   ├── index.html           # Home / Predict / About
│   ├── style.css
│   └── script.js
└── README.md
```

## Getting started

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and a modern browser. No local Python install is required — everything runs inside the container.

**1. Clone the repository**

```bash
git clone https://github.com/AbdulHannan-Offixial/student-performance-predictor.git
cd student-performance-predictor
```

**2. Start the backend**

```bash
docker compose up --build
```

This builds the image, installs dependencies, and starts the API at `http://localhost:8000`. First build takes a few minutes; later ones are much faster.

**3. Check it's running**

Open `http://localhost:8000/docs` — FastAPI's interactive docs. Expand `POST /predict`, click "Try it out," and execute the example request to confirm you get a score back.

**4. Open the frontend**

Open `frontend/index.html` directly in a browser (no server needed). It talks to the backend at `http://localhost:8000` by default — update the `API_BASE_URL` constant at the top of `script.js` once the backend is deployed somewhere other than your own machine.

**5. Stop it when you're done**

`Ctrl+C` in the terminal running the container, or `docker compose down` from another terminal.

## Retraining the model

```bash
cd backend/model
python train_model.py
```

Loads `data/student_data.csv`, splits it 80/20 into train and test sets, trains both candidate models, prints their comparison metrics, and overwrites `student_model.pkl` with whichever generalizes better.

## API reference

**`POST /predict`**

Request body (all 19 fields required):

```json
{
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
  "Gender": "Male"
}
```

Response:

```json
{
  "predicted_exam_score": 68.99
}
```

Full, explorable documentation — including every field's valid values and ranges — is auto-generated at `/docs` while the server is running.

## Limitations & possible next steps

- The model is a single linear regression fit on one dataset; it hasn't been validated against real classroom data outside of Kaggle's sample.
- The API has no authentication or rate limiting, which is fine for a demo but would need addressing before any public deployment handling real student data.
- Predictions aren't logged anywhere, so there's no way to monitor accuracy drift over time.
- Trying gradient-boosted models (XGBoost, LightGBM) or feature engineering (interaction terms between attendance and tutoring, for example) could be a natural next experiment.

## Author

**Abdul Hannan** — AI Engineer
This project was built end-to-end as a learning exercise: FastAPI, scikit-learn, and Docker, trained on a student performance dataset from Kaggle.

[GitHub](https://github.com/AbdulHannan-Offixial) · [LinkedIn](https://www.linkedin.com/in/abdul-hannan-9188b62b2)
