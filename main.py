from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Store the loaded model and threshold
ml_model = {}

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load(BASE_DIR / "credit_risk_model.pkl")

    threshold_value = joblib.load(BASE_DIR / "best_threshold.pkl")
    ml_model["threshold"] = float(threshold_value)

    yield

    ml_model.clear()


app = FastAPI(lifespan=lifespan)


# Input validation
class LoanApplication(BaseModel):
    person_age: int
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
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.model_dump()])

    # Predict probability and convert NumPy value to Python float
    probability = float(
        ml_model["model"].predict_proba(input_df)[0, 1]
    )

    threshold = float(ml_model["threshold"])
    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }


# Serve the frontend
app.mount(
    "/",
    StaticFiles(directory=str(BASE_DIR / "static"), html=True),
    name="static"
)
