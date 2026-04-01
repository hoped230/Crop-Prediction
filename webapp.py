from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "research_run" / "best_model" / "best_pipeline.joblib"
PROFILE_PATH = BASE_DIR / "artifacts" / "research_run" / "best_model" / "data_profile.json"
METADATA_PATH = BASE_DIR / "artifacts" / "research_run" / "best_model" / "best_model_metadata.json"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CROP_PRESETS = [
    {"name": "Rice", "emoji": "🌾", "season": "Kharif", "N": 80, "P": 40, "K": 40, "temperature": 25, "humidity": 82, "ph": 6.5, "rainfall": 200},
    {"name": "Maize", "emoji": "🌽", "season": "Rabi", "N": 77, "P": 48, "K": 20, "temperature": 22, "humidity": 65, "ph": 6.0, "rainfall": 65},
    {"name": "Sugarcane", "emoji": "🎋", "season": "Annual", "N": 120, "P": 36, "K": 38, "temperature": 28, "humidity": 78, "ph": 6.5, "rainfall": 145},
    {"name": "Cotton", "emoji": "🌸", "season": "Kharif", "N": 117, "P": 47, "K": 19, "temperature": 24, "humidity": 79, "ph": 7.0, "rainfall": 80},
    {"name": "Coffee", "emoji": "☕", "season": "Perennial", "N": 101, "P": 28, "K": 30, "temperature": 25, "humidity": 58, "ph": 7.2, "rainfall": 150},
    {"name": "Coconut", "emoji": "🥥", "season": "Perennial", "N": 22, "P": 16, "K": 30, "temperature": 27, "humidity": 92, "ph": 5.9, "rainfall": 172},
    {"name": "Mango", "emoji": "🥭", "season": "Summer", "N": 18, "P": 22, "K": 20, "temperature": 30, "humidity": 50, "ph": 5.5, "rainfall": 90},
    {"name": "Jute", "emoji": "🌿", "season": "Kharif", "N": 78, "P": 46, "K": 39, "temperature": 33, "humidity": 80, "ph": 6.0, "rainfall": 175},
    {"name": "Lentil", "emoji": "🫘", "season": "Rabi", "N": 18, "P": 68, "K": 19, "temperature": 24, "humidity": 65, "ph": 6.6, "rainfall": 45},
    {"name": "Banana", "emoji": "🍌", "season": "Annual", "N": 100, "P": 82, "K": 50, "temperature": 27, "humidity": 80, "ph": 6.0, "rainfall": 105},
    {"name": "Papaya", "emoji": "🍈", "season": "Annual", "N": 50, "P": 60, "K": 55, "temperature": 33, "humidity": 92, "ph": 6.5, "rainfall": 145},
    {"name": "Grapes", "emoji": "🍇", "season": "Perennial", "N": 23, "P": 132, "K": 200, "temperature": 23, "humidity": 82, "ph": 6.0, "rainfall": 70},
]

FORECAST = [
    {"day": "Wed", "temp": 32, "rain": 15},
    {"day": "Thu", "temp": 31, "rain": 35},
    {"day": "Fri", "temp": 32, "rain": 50},
    {"day": "Sat", "temp": 32, "rain": 65},
    {"day": "Sun", "temp": 32, "rain": 5},
]


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
model = joblib.load(MODEL_PATH)
profile = load_json(PROFILE_PATH)
metadata = load_json(METADATA_PATH)
preset_lookup = {item["name"].lower(): item for item in CROP_PRESETS}


def default_value(name: str, fallback: float) -> float:
    return float(profile.get(name, {}).get("median", fallback))


@app.get("/")
def index():
    defaults = {
        "N": default_value("N", 90),
        "P": default_value("P", 42),
        "K": default_value("K", 43),
        "temperature": default_value("temperature", 32),
        "humidity": default_value("humidity", 72),
        "ph": default_value("ph", 6.5),
        "rainfall": default_value("rainfall", 202),
    }
    return render_template(
        "index.html",
        metadata=metadata,
        defaults=defaults,
        crops=CROP_PRESETS,
        forecast=FORECAST,
        crop_count=len(getattr(model, "classes_", [])) or 22,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    values = {feature: float(payload[feature]) for feature in FEATURES}
    input_frame = pd.DataFrame([values])
    prediction = model.predict(input_frame)[0]
    response = {
        "prediction": prediction,
        "details": preset_lookup.get(prediction.lower(), {"emoji": "🌱", "season": "Recommended"}),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)[0]
        ranked = sorted(
            [{"crop": cls, "confidence": float(prob)} for cls, prob in zip(model.classes_, probabilities)],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        for item in ranked:
            item["confidence_pct"] = round(item["confidence"] * 100, 2)
        response["top_predictions"] = ranked[:5]

    return jsonify(response)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
