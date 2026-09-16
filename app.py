"""
Heart Disease Risk Prediction - Flask backend.

Loads the pre-trained Logistic Regression model (model/logistic_regression_heart.pkl)
together with the exact preprocessing pipeline it was trained with:

  1. Zero-value cleaning for Cholesterol / RestingBP (0 -> mean of non-zero values,
     reproduced from the training notebook using the original heart.csv).
  2. One-hot encoding (pd.get_dummies(drop_first=True)) matching model/columns.pkl.
  3. A first-stage StandardScaler fit on the numerical columns of the FULL dataset
     (this reproduces a scaling step baked into the training notebook before the
     train/test split - it is not saved as a .pkl, so it is recomputed at startup
     from heart.csv, which is deterministic).
  4. The saved second-stage StandardScaler (model/sclaer.pkl), applied to all 15
     features exactly as during training.

No retraining happens here - only the existing model.pkl / scaler.pkl are used.
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")

NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

CHEST_PAIN_TYPES = {
    "ASY": "Asymptomatic",
    "ATA": "Atypical Angina",
    "NAP": "Non-Anginal Pain",
    "TA": "Typical Angina",
}
RESTING_ECG_TYPES = {
    "Normal": "Normal",
    "ST": "ST-T Wave Abnormality",
    "LVH": "Left Ventricular Hypertrophy",
}
ST_SLOPE_TYPES = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_disease_app")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24).hex())


class ModelBundle:
    """Holds the trained model plus everything needed to reproduce training preprocessing."""

    def __init__(self):
        logger.info("Loading model artifacts from %s", MODEL_DIR)
        self.model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_heart.pkl"))
        self.final_scaler = joblib.load(os.path.join(MODEL_DIR, "sclaer.pkl"))
        self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))

        raw = pd.read_csv(DATA_PATH)

        self.cholesterol_fill = round(
            raw.loc[raw["Cholesterol"] != 0, "Cholesterol"].mean(), 2
        )
        self.resting_bp_fill = round(
            raw.loc[raw["RestingBP"] != 0, "RestingBP"].mean(), 2
        )

        cleaned = raw.copy()
        cleaned["Cholesterol"] = cleaned["Cholesterol"].replace(0, self.cholesterol_fill).round(2)
        cleaned["RestingBP"] = cleaned["RestingBP"].replace(0, self.resting_bp_fill).round(2)

        encoded = pd.get_dummies(cleaned, drop_first=True).astype(int)
        self.pre_scaler = StandardScaler().fit(encoded[NUMERICAL_COLS])

        logger.info("Model, scaler and %d training columns loaded successfully.", len(self.feature_columns))

    def build_feature_row(self, form):
        row = {
            "Age": form["age"],
            "RestingBP": form["resting_bp"],
            "Cholesterol": form["cholesterol"],
            "FastingBS": form["fasting_bs"],
            "MaxHR": form["max_hr"],
            "Oldpeak": form["oldpeak"],
            "Sex_M": 1 if form["sex"] == "M" else 0,
            "ChestPainType_ATA": 1 if form["chest_pain_type"] == "ATA" else 0,
            "ChestPainType_NAP": 1 if form["chest_pain_type"] == "NAP" else 0,
            "ChestPainType_TA": 1 if form["chest_pain_type"] == "TA" else 0,
            "RestingECG_Normal": 1 if form["resting_ecg"] == "Normal" else 0,
            "RestingECG_ST": 1 if form["resting_ecg"] == "ST" else 0,
            "ExerciseAngina_Y": 1 if form["exercise_angina"] == "Y" else 0,
            "ST_Slope_Flat": 1 if form["st_slope"] == "Flat" else 0,
            "ST_Slope_Up": 1 if form["st_slope"] == "Up" else 0,
        }

        if row["Cholesterol"] == 0:
            row["Cholesterol"] = self.cholesterol_fill
        if row["RestingBP"] == 0:
            row["RestingBP"] = self.resting_bp_fill

        df = pd.DataFrame([row], columns=self.feature_columns)
        df[NUMERICAL_COLS] = self.pre_scaler.transform(df[NUMERICAL_COLS])
        scaled = self.final_scaler.transform(df)
        return scaled

    def predict(self, form):
        features = self.build_feature_row(form)
        prediction = int(self.model.predict(features)[0])
        probability = None
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(features)[0][1])
        return prediction, probability


bundle = ModelBundle()


class ValidationError(Exception):
    pass


def _to_int(value, field, min_value, max_value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a whole number.")
    if not (min_value <= parsed <= max_value):
        raise ValidationError(f"{field} must be between {min_value} and {max_value}.")
    return parsed


def _to_float(value, field, min_value, max_value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number.")
    if not (min_value <= parsed <= max_value):
        raise ValidationError(f"{field} must be between {min_value} and {max_value}.")
    return parsed


def _to_choice(value, field, allowed):
    if value not in allowed:
        raise ValidationError(f"{field} has an invalid value.")
    return value


def validate_form(raw):
    data = {}
    data["age"] = _to_int(raw.get("age"), "Age", 1, 120)
    data["sex"] = _to_choice(raw.get("sex"), "Sex", {"M", "F"})
    data["chest_pain_type"] = _to_choice(
        raw.get("chest_pain_type"), "Chest pain type", set(CHEST_PAIN_TYPES)
    )
    data["resting_bp"] = _to_int(raw.get("resting_bp"), "Resting blood pressure", 0, 250)
    data["cholesterol"] = _to_int(raw.get("cholesterol"), "Cholesterol", 0, 700)
    data["fasting_bs"] = _to_choice(raw.get("fasting_bs"), "Fasting blood sugar", {"0", "1"})
    data["fasting_bs"] = int(data["fasting_bs"])
    data["resting_ecg"] = _to_choice(
        raw.get("resting_ecg"), "Resting ECG", set(RESTING_ECG_TYPES)
    )
    data["max_hr"] = _to_int(raw.get("max_hr"), "Maximum heart rate", 60, 220)
    data["exercise_angina"] = _to_choice(
        raw.get("exercise_angina"), "Exercise-induced angina", {"Y", "N"}
    )
    data["oldpeak"] = _to_float(raw.get("oldpeak"), "Oldpeak", -3.0, 7.0)
    data["st_slope"] = _to_choice(raw.get("st_slope"), "ST slope", set(ST_SLOPE_TYPES))
    return data


@app.route("/")
def index():
    return render_template(
        "index.html",
        chest_pain_types=CHEST_PAIN_TYPES,
        resting_ecg_types=RESTING_ECG_TYPES,
        st_slope_types=ST_SLOPE_TYPES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    context = {
        "chest_pain_types": CHEST_PAIN_TYPES,
        "resting_ecg_types": RESTING_ECG_TYPES,
        "st_slope_types": ST_SLOPE_TYPES,
    }
    try:
        data = validate_form(request.form)
    except ValidationError as exc:
        return render_template(
            "index.html",
            error=str(exc),
            form=request.form,
            **context,
        ), 400

    try:
        prediction, probability = bundle.predict(data)
    except Exception:
        logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Unable to generate prediction. Please verify the entered information and try again.",
            form=request.form,
            **context,
        ), 500

    summary = [
        ("Age", data["age"]),
        ("Sex", "Male" if data["sex"] == "M" else "Female"),
        ("Chest Pain Type", CHEST_PAIN_TYPES[data["chest_pain_type"]]),
        ("Resting Blood Pressure", f"{data['resting_bp']} mm Hg"),
        ("Cholesterol", f"{data['cholesterol']} mg/dl"),
        ("Fasting Blood Sugar > 120 mg/dl", "Yes" if data["fasting_bs"] == 1 else "No"),
        ("Resting ECG", RESTING_ECG_TYPES[data["resting_ecg"]]),
        ("Maximum Heart Rate", data["max_hr"]),
        ("Exercise-Induced Angina", "Yes" if data["exercise_angina"] == "Y" else "No"),
        ("Oldpeak (ST Depression)", data["oldpeak"]),
        ("ST Slope", ST_SLOPE_TYPES[data["st_slope"]]),
    ]

    return render_template(
        "result.html",
        prediction=prediction,
        probability=probability,
        summary=summary,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
