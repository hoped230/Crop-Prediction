from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "research_run" / "best_model" / "best_pipeline.joblib"
PROFILE_PATH = BASE_DIR / "artifacts" / "research_run" / "best_model" / "data_profile.json"
METADATA_PATH = BASE_DIR / "artifacts" / "research_run" / "best_model" / "best_model_metadata.json"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

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


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict | list:
    request_obj = Request(url, headers=headers or {})
    with urlopen(request_obj, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


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


@app.get("/api/weather")
def weather():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    if not OPENWEATHER_API_KEY:
        return jsonify({"error": "OpenWeather API key is not configured"}), 500

    current_query = urlencode(
        {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }
    )
    forecast_query = urlencode(
        {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "cnt": 40,
        }
    )
    reverse_query = urlencode({"lat": lat, "lon": lon, "limit": 1, "appid": OPENWEATHER_API_KEY})

    def fetch_openweather(url: str, required: bool = True) -> dict | list:
        try:
            return fetch_json(url)
        except HTTPError as exc:
            details = ""
            try:
                error_body = json.loads(exc.read().decode("utf-8"))
                details = error_body.get("message", "")
            except Exception:
                details = ""
            if required:
                message = f"OpenWeather request failed with status {exc.code}"
                if details:
                    message = f"{message}: {details}"
                raise RuntimeError(message) from exc
            return {}
        except URLError as exc:
            if required:
                raise RuntimeError(f"Unable to reach OpenWeather: {exc.reason}") from exc
            return {}

    try:
        current = fetch_openweather(f"https://api.openweathermap.org/data/2.5/weather?{current_query}")
        forecast = fetch_openweather(f"https://api.openweathermap.org/data/2.5/forecast?{forecast_query}")
        reverse = fetch_openweather(f"https://api.openweathermap.org/geo/1.0/reverse?{reverse_query}", required=False)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502

    grouped_days: dict[str, dict] = {}
    for item in forecast.get("list", []):
        date_key = item.get("dt_txt", "")[:10]
        if not date_key:
            continue

        main = item.get("main", {})
        weather_items = item.get("weather") or [{}]
        rain_amount = float((item.get("rain") or {}).get("3h", 0.0) or 0.0)
        pop = float(item.get("pop", 0.0) or 0.0)

        day_entry = grouped_days.setdefault(
            date_key,
            {
                "date": date_key,
                "temp_values": [],
                "rain_mm": 0.0,
                "pop_values": [],
                "descriptions": [],
                "icons": [],
            },
        )
        if isinstance(main.get("temp_max"), (int, float)):
            day_entry["temp_values"].append(float(main["temp_max"]))
        elif isinstance(main.get("temp"), (int, float)):
            day_entry["temp_values"].append(float(main["temp"]))
        day_entry["rain_mm"] += rain_amount
        day_entry["pop_values"].append(pop)
        if weather_items[0].get("description"):
            day_entry["descriptions"].append(weather_items[0]["description"])
        if weather_items[0].get("icon"):
            day_entry["icons"].append(weather_items[0]["icon"])

    daily_forecast = []
    for _, item in list(grouped_days.items())[:5]:
        max_temp = max(item["temp_values"]) if item["temp_values"] else None
        rain_probability = round(max(item["pop_values"]) * 100) if item["pop_values"] else 0
        daily_forecast.append(
            {
                "date": item["date"],
                "temp": round(max_temp) if max_temp is not None else None,
                "rain": rain_probability,
                "rain_mm": round(item["rain_mm"], 1),
                "description": item["descriptions"][0] if item["descriptions"] else "",
                "icon": item["icons"][0] if item["icons"] else "",
            }
        )

    city_data = reverse[0] if isinstance(reverse, list) and reverse else {}
    current_weather = (current.get("weather") or [{}])[0]
    current_main = current.get("main", {})

    payload = {
        "location": {
            "name": city_data.get("name") or current.get("name") or "Current location",
            "state": city_data.get("state") or "",
            "country": city_data.get("country") or current.get("sys", {}).get("country") or "",
        },
        "current": {
            "temp": current_main.get("temp"),
            "humidity": current_main.get("humidity"),
            "description": current_weather.get("description", ""),
            "icon": current_weather.get("icon", ""),
            "timestamp": current.get("dt"),
        },
        "forecast": daily_forecast,
    }
    return jsonify(payload)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
