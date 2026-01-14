from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
    "day"
]


MODEL_PATH = "models/hgb_pipeline.joblib"


# Load model once at startup
MODEL = joblib.load(MODEL_PATH)



def to_bool(value: str) -> bool:
    """Convert 'yes/no' or 'true/false' to bool."""
    v = value.strip().lower()
    return v in ["yes", "true", "1", "y"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Read user-friendly inputs from form
            season = int(request.form["season"])
            yr = int(request.form["yr"])
            mnth = int(request.form["mnth"])
            day = int(request.form["day"])
            hr = int(request.form["hr"])
            weekday = int(request.form["weekday"])
            weathersit = int(request.form["weathersit"])

            holiday = 1 if to_bool(request.form["holiday"]) else 0
            workingday = 1 if to_bool(request.form["workingday"]) else 0

            # User-friendly units
            temp_c = float(request.form["temp_c"])
            hum_percent = float(request.form["hum_percent"])
            wind = float(request.form["windspeed"])

            # Convert to dataset-normalized features
            features = {
                "season": season,
                "yr": yr,
                "mnth": mnth,
                "hr": hr,
                "holiday": holiday,
                "workingday": workingday,
                "weekday": weekday,
                "weathersit": weathersit,
                "temp": temp_c / 41.0,
                "hum": hum_percent / 100.0,
                "windspeed": wind / 67.0,
                "day": day,
            }

            # Build dataframe 
            X = pd.DataFrame([features], columns=FEATURES)

            # Predict
            y_pred = MODEL.predict(X)[0]
            prediction = float(y_pred)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error, form=request.form)

if __name__ == "__main__":
    # Run local web server
    app.run(host="127.0.0.1", port=5001, debug=True)


