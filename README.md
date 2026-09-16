# Heart Disease Prediction using Machine Learning

**Live demo:** https://heart-disease-prediction-mmd2.onrender.com
*(hosted on Render's free tier - may take 30-50s to wake up after idling)*

## Overview

This project provides a web interface for predicting the likelihood of heart disease
from a patient's clinical measurements, backed by a pre-trained scikit-learn model.

> **Note:** the dataset used (`heart.csv`, the UCI Heart Failure / Heart Disease
> dataset) predicts **heart disease** (target column `HeartDisease`), not stroke.
> The UI and README reflect this so predictions are described accurately.

## Machine Learning Model

Several classification algorithms (Logistic Regression, KNN, Naive Bayes, Decision
Tree, SVM) were evaluated on the dataset during model development (see
`Heart Stroke.ipynb`). Logistic Regression was selected for this application based on
the evaluation performed in that notebook. This is not a claim that it is universally
the best algorithm for this problem — only that it performed well for this dataset
and evaluation setup.

The model, its `StandardScaler`, and the exact training feature order are loaded
from `model/logistic_regression_heart.pkl`, `model/sclaer.pkl`, and
`model/columns.pkl` respectively. **The model is not retrained by this application.**

## Technologies

* Python
* Flask
* Scikit-learn
* Pandas
* HTML / CSS / JavaScript (Bootstrap 5)

## Features

* Patient data input form organized into Personal, Health, and Exercise/Lifestyle sections
* ML-based prediction with model probability
* Responsive UI (desktop, tablet, mobile)
* Frontend and backend input validation
* Friendly error handling (no tracebacks shown to users)

## Installation

Create a virtual environment:

```
python -m venv venv
```

Activate it (Windows):

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Deployment (Render)

This repo includes a `render.yaml` so Render can deploy it with no manual
configuration:

1. Push the repo to GitHub (already done if you're reading this on GitHub).
2. Go to [render.com](https://render.com) → **New** → **Blueprint**, and select
   this repository. Render reads `render.yaml` and configures the service
   automatically (build command `pip install -r requirements.txt`, start
   command `gunicorn app:app`).
3. Deploy. Render gives you a public URL such as
   `https://heart-disease-prediction.onrender.com`.

Every push to `main` triggers an automatic redeploy. Note: on Render's free
plan the service spins down after ~15 minutes of inactivity, so the first
request after idling takes 30-50 seconds to wake up.

To deploy without the blueprint (manual setup), create a **Web Service**
pointing at this repo with:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

## Project Structure

```
heart-stroke project/
│
├── app.py                 # Flask app: routes, validation, preprocessing, prediction
├── requirements.txt
├── heart.csv               # Original training dataset (used to reproduce preprocessing)
├── Heart Stroke.ipynb       # Original model training / EDA notebook
│
├── model/
│   ├── logistic_regression_heart.pkl   # Trained Logistic Regression model
│   ├── sclaer.pkl                      # StandardScaler fit during training
│   └── columns.pkl                     # Exact feature column order used in training
│
├── templates/
│   ├── index.html          # Prediction form
│   └── result.html         # Prediction result page
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
└── README.md
```

## How Preprocessing Works

The trained pipeline (from `Heart Stroke.ipynb`) is reproduced exactly at inference
time:

1. `Cholesterol` and `RestingBP` values of `0` are replaced with the mean of the
   non-zero values from `heart.csv` (recomputed at startup, matching training).
2. Categorical fields (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`,
   `ST_Slope`) are one-hot encoded to match `model/columns.pkl` exactly
   (`pd.get_dummies(drop_first=True)`).
3. The five numerical columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`,
   `Oldpeak`) pass through a `StandardScaler` fit on the full `heart.csv` dataset —
   this reproduces a scaling step that happened in the training notebook before the
   train/test split, and is recomputed at startup since it wasn't saved as a `.pkl`.
4. All 15 resulting features then pass through the saved `model/sclaer.pkl`
   `StandardScaler`, exactly as during training.
5. The scaled features are passed to `model/logistic_regression_heart.pkl` for
   `.predict()` and `.predict_proba()`.

## Medical Disclaimer

This tool is intended for educational and research purposes only. It is not a
medical diagnostic tool and should not replace professional medical advice. If you
have concerns about your heart health, consult a qualified healthcare professional.
