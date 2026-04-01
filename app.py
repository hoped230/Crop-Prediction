from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


MODEL_PATH = Path("artifacts/research_run/best_model/best_pipeline.joblib")
PROFILE_PATH = Path("artifacts/research_run/best_model/data_profile.json")
METADATA_PATH = Path("artifacts/research_run/best_model/best_model_metadata.json")

CROP_PRESETS = [
    {"name": "Rice", "emoji": "🌾", "season": "Kharif", "N": 80, "P": 40, "K": 40, "temperature": 25, "humidity": 82, "ph": 6.5, "rainfall": 200},
    {"name": "Maize", "emoji": "🌽", "season": "Rabi", "N": 77, "P": 48, "K": 20, "temperature": 22, "humidity": 65, "ph": 6.0, "rainfall": 65},
    {"name": "Cotton", "emoji": "🌸", "season": "Kharif", "N": 117, "P": 47, "K": 19, "temperature": 24, "humidity": 79, "ph": 7.0, "rainfall": 80},
    {"name": "Coffee", "emoji": "☕", "season": "Perennial", "N": 101, "P": 28, "K": 30, "temperature": 25, "humidity": 58, "ph": 7.2, "rainfall": 150},
    {"name": "Coconut", "emoji": "🥥", "season": "Perennial", "N": 22, "P": 16, "K": 30, "temperature": 27, "humidity": 92, "ph": 5.9, "rainfall": 172},
    {"name": "Banana", "emoji": "🍌", "season": "Annual", "N": 100, "P": 82, "K": 50, "temperature": 27, "humidity": 80, "ph": 6.0, "rainfall": 105},
    {"name": "Jute", "emoji": "🌿", "season": "Kharif", "N": 78, "P": 46, "K": 39, "temperature": 33, "humidity": 80, "ph": 6.0, "rainfall": 175},
    {"name": "Grapes", "emoji": "🍇", "season": "Perennial", "N": 23, "P": 132, "K": 200, "temperature": 23, "humidity": 82, "ph": 6.0, "rainfall": 70},
]

FORECAST = [
    {"day": "Wed", "temp": 32, "rain": 15},
    {"day": "Thu", "temp": 31, "rain": 35},
    {"day": "Fri", "temp": 32, "rain": 50},
    {"day": "Sat", "temp": 32, "rain": 65},
    {"day": "Sun", "temp": 32, "rain": 5},
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def svg_data_uri(svg: str) -> str:
    return f"data:image/svg+xml;utf8,{quote(svg)}"


def field_scene() -> str:
    return svg_data_uri(
        """
        <svg viewBox="0 0 700 185" xmlns="http://www.w3.org/2000/svg">
          <defs><linearGradient id="skyG" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#89d4f0" stop-opacity="0.5"/><stop offset="100%" stop-color="#c8f0b0" stop-opacity="0.3"/>
          </linearGradient></defs>
          <rect width="700" height="185" fill="url(#skyG)" rx="16"/>
          <circle cx="600" cy="44" r="30" fill="#ffd166" opacity="0.9"/>
          <path d="M0 102 C80 72, 162 67, 242 92 S362 118, 482 87 S602 67, 700 82 L700 185 L0 185 Z" fill="#4d9f65" opacity="0.80"/>
          <path d="M0 122 C102 97, 202 102, 302 120 S452 137, 602 112 L700 117 L700 185 L0 185 Z" fill="#3a7a52"/>
          <rect x="0" y="148" width="700" height="37" fill="#5c4026"/>
        </svg>
        """
    )


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: linear-gradient(160deg, #e8f5e9 0%, #f0faf2 35%, #fffde7 70%, #fff8f0 100%); }
        .block-container { max-width: 1120px; padding-top: 1rem; padding-bottom: 2.5rem; }
        h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #1b2e1e; }
        .wrap { background: rgba(255,255,255,0.82); border: 1px solid rgba(45,106,79,0.10); box-shadow: 0 10px 38px rgba(27,46,30,0.07); backdrop-filter: blur(14px); border-radius: 26px; padding: 24px; }
        .top { display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; }
        .brand { display:flex; align-items:center; gap:12px; }
        .icon { width:44px; height:44px; border-radius:13px; background:linear-gradient(135deg, #2d6a4f, #52b788); display:flex; align-items:center; justify-content:center; font-size:22px; }
        .brand-name { font-family:'Playfair Display', serif; font-size:28px; font-weight:900; color:#1b2e1e; }
        .brand-sub { font-size:12px; color:#5a7c65; }
        .pillrow { display:flex; gap:10px; }
        .pill { background: rgba(255,255,255,0.72); border: 1px solid rgba(45,106,79,0.15); border-radius: 50px; padding: 8px 16px; font-size: 13px; }
        .active { background:#2d6a4f; color:white; border-color:#2d6a4f; }
        .hero { background: linear-gradient(140deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%); border-radius:30px; padding:34px; color:white; box-shadow:0 22px 64px rgba(27,67,50,0.30); }
        .tag { font-size:11px; letter-spacing:2.5px; text-transform:uppercase; opacity:0.68; margin-bottom:10px; }
        .title { font-family:'Playfair Display', serif; font-size:44px; line-height:1.08; font-weight:900; color:white; margin-bottom:16px; }
        .title span { color:#95d5b2; }
        .desc { font-size:14px; line-height:1.7; opacity:0.86; margin-bottom:18px; }
        .badges { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
        .badge { background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.22); border-radius:50px; padding:6px 14px; font-size:12px; }
        .weather { background: linear-gradient(150deg, #023e8a 0%, #0077b6 45%, #0096c7 80%, #00b4d8 100%); border-radius:30px; padding:26px; color:white; box-shadow:0 22px 54px rgba(0,119,182,0.32); }
        .weather-small { font-size:11px; letter-spacing:2px; text-transform:uppercase; opacity:0.72; }
        .weather-city { font-size:19px; font-weight:600; margin-top:4px; }
        .weather-temp { font-family:'Playfair Display', serif; font-size:54px; font-weight:900; line-height:1; margin-top:18px; }
        .weather-sub { font-size:12px; opacity:0.72; margin-top:8px; }
        .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:15px; margin:1.2rem 0; }
        .stat { background: rgba(255,255,255,0.82); border-radius:18px; padding:18px 20px; border:1px solid rgba(45,106,79,0.10); }
        .slabel { font-size:11px; text-transform:uppercase; letter-spacing:1.6px; color:#5a7c65; }
        .sval { font-size:30px; font-weight:700; color:#1b2e1e; line-height:1.1; margin-top:4px; }
        .ssub { font-size:12px; color:#5a7c65; margin-top:4px; }
        .ctitle { font-family:'Playfair Display', serif; font-size:20px; font-weight:700; color:#1b2e1e; margin-bottom:4px; }
        .csub { font-size:13px; color:#5a7c65; margin-bottom:18px; }
        .result { background: linear-gradient(135deg, #fff9db, #ffeaa7); border:1.5px solid #f4c842; border-radius:20px; padding:18px 22px; margin-top:18px; }
        .rlabel { font-size:11px; text-transform:uppercase; letter-spacing:1.6px; color:#8c6c00; font-weight:600; }
        .rcrop { font-family:'Playfair Display', serif; font-size:34px; font-weight:900; color:#3d3203; margin:5px 0 7px; }
        .rdesc { font-size:13px; color:#5c4700; line-height:1.65; }
        .chip { background: rgba(255,255,255,0.82); border-radius:18px; padding:10px; text-align:center; border:1px solid rgba(45,106,79,0.10); margin-top:8px; }
        .chip-name { font-size:12px; font-weight:700; color:#1b2e1e; }
        .chip-season { font-size:10px; color:#5a7c65; }
        div[data-testid="stForm"] { border:none !important; background:transparent !important; padding:0 !important; }
        @media (max-width: 900px) { .stats { grid-template-columns:1fr 1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state(profile: dict):
    defaults = {
        "N": float(profile.get("N", {}).get("median", 90)),
        "P": float(profile.get("P", {}).get("median", 42)),
        "K": float(profile.get("K", {}).get("median", 43)),
        "temperature": float(profile.get("temperature", {}).get("median", 32)),
        "humidity": float(profile.get("humidity", {}).get("median", 72)),
        "ph": float(profile.get("ph", {}).get("median", 6.5)),
        "rainfall": float(profile.get("rainfall", {}).get("median", 202)),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(preset: dict):
    for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        st.session_state[key] = float(preset[key])


def input_df() -> pd.DataFrame:
    return pd.DataFrame([{k: float(st.session_state[k]) for k in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]}])


def radar_chart():
    vals = [min(st.session_state["N"], 140), min(st.session_state["P"], 145), min(st.session_state["K"], 205), st.session_state["ph"] * 15, st.session_state["temperature"] * 3, st.session_state["humidity"]]
    labels = ["N", "P", "K", "pH×15", "Temp×3", "Humidity"]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself", line=dict(color="#52b788", width=3), fillcolor="rgba(82,183,136,0.22)"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(showticklabels=False)))
    return fig


def npk_chart():
    fig = go.Figure(go.Bar(x=["N", "P", "K"], y=[st.session_state["N"], st.session_state["P"], st.session_state["K"]], marker_color=["#52b788", "#e9c46a", "#f4a261"]))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def weather_chart(kind: str):
    if kind == "rain":
        fig = go.Figure(go.Bar(x=[x["day"] for x in FORECAST], y=[x["rain"] for x in FORECAST], marker_color=["#90e0ef", "#48cae4", "#0096c7", "#0077b6", "#caf0f8"]))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        return fig
    fig = go.Figure(go.Scatter(x=[x["day"] for x in FORECAST], y=[x["temp"] for x in FORECAST], mode="lines+markers", line=dict(color="#f4a261", width=3, shape="spline"), fill="tozeroy", fillcolor="rgba(244,162,97,0.14)"))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def render_result(model):
    df = input_df()
    prediction = model.predict(df)[0]
    probs = model.predict_proba(df)[0] if hasattr(model, "predict_proba") else None
    lookup = {item["name"].lower(): item for item in CROP_PRESETS}
    chosen = lookup.get(prediction.lower(), {"emoji": "🌱", "season": "Recommended"})
    st.markdown(f'<div class="result"><div class="rlabel">Best Recommendation</div><div class="rcrop">{chosen["emoji"]} {prediction}</div><div class="rdesc">Optimal match for your soil and climate profile. {chosen["season"]} crop from the trained model output.</div></div>', unsafe_allow_html=True)
    if probs is not None:
        result_df = pd.DataFrame({"Crop": model.classes_, "Confidence": probs}).sort_values("Confidence", ascending=False)
        result_df["Confidence %"] = (result_df["Confidence"] * 100).round(2)
        c1, c2 = st.columns([0.95, 1.05], gap="large")
        with c1:
            st.markdown('<div class="wrap"><div class="ctitle">📋 Prediction Summary</div><div class="csub">Top-ranked crops from the model</div>', unsafe_allow_html=True)
            st.dataframe(result_df.head(5)[["Crop", "Confidence %"]], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="wrap"><div class="ctitle">📊 Confidence Distribution</div><div class="csub">Highest-probability crops for this field</div>', unsafe_allow_html=True)
            st.bar_chart(result_df.head(5).set_index("Crop")[["Confidence"]], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="CropWise — Smart Agronomic Intelligence", page_icon="🌿", layout="wide")
    inject_styles()

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python run_research_pipeline.py --data data\\Crop_recommendation.csv` first.")
        return

    model = load_model()
    profile = load_json(PROFILE_PATH)
    meta = load_json(METADATA_PATH)
    metrics = meta.get("metrics", {})
    init_state(profile)

    st.markdown('<div class="top"><div class="brand"><div class="icon">🌿</div><div><div class="brand-name">CropWise</div><div class="brand-sub">Smart Agronomic Intelligence</div></div></div><div class="pillrow"><div class="pill active">Dashboard</div><div class="pill">Analytics</div><div class="pill">Fields</div></div></div>', unsafe_allow_html=True)

    h1, h2 = st.columns([1.15, 0.85], gap="large")
    with h1:
        st.markdown(f'<div class="hero"><div class="tag">AI-Powered Crop Intelligence</div><div class="title">Choose a better<br/><span>crop</span>, yield more</div><div class="desc">Your UI design is now connected to the trained crop recommendation model. Enter field values below and get real predictions with confidence scores.</div><div class="badges"><span class="badge">📊 {meta.get("best_model", "Model")} Active</span><span class="badge">🌾 {len(getattr(model, "classes_", [])) or 22} Crop Types</span><span class="badge">🧪 Soil + Weather Inputs</span></div><img src="{field_scene()}" style="width:100%;border-radius:20px;margin-top:6px;" alt="field"/></div>', unsafe_allow_html=True)
    with h2:
        forecast_boxes = "".join([f'<div class="pill" style="background:rgba(255,255,255,0.13);color:white;border-color:rgba(255,255,255,0.16);text-align:center;flex:1;"><div style="font-size:10px;opacity:0.72;">{x["day"]}</div><div style="font-size:14px;font-weight:700;">{x["temp"]}°</div><div style="font-size:10px;opacity:0.72;">💧 {x["rain"]}%</div></div>' for x in FORECAST])
        st.markdown(f'<div class="weather"><div class="weather-small">Current Weather</div><div class="weather-city">Detected Conditions</div><div class="weather-sub">Smart local conditions preview</div><div class="weather-temp">{st.session_state["temperature"]:.1f}°C</div><div class="weather-sub">Humidity: ~{st.session_state["humidity"]:.1f}% | Rainfall input: {st.session_state["rainfall"]:.1f} mm</div><div class="weather-sub" style="margin-top:14px;margin-bottom:8px;letter-spacing:1.6px;text-transform:uppercase;">5-Day Forecast</div><div style="display:flex;gap:7px;">{forecast_boxes}</div><div class="weather-sub" style="margin-top:14px;">🌾 Tip: this recommendation comes from the trained model, not hardcoded crop rules.</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="stats"><div class="stat"><div class="slabel">Model Accuracy</div><div class="sval">{metrics.get("accuracy",0)*100:.1f}%</div><div class="ssub">Top performing</div></div><div class="stat"><div class="slabel">Crops Supported</div><div class="sval">{len(getattr(model, "classes_", [])) or 22}</div><div class="ssub">Multi-class recommendation</div></div><div class="stat"><div class="slabel">Field Temp</div><div class="sval">{st.session_state["temperature"]:.1f}°C</div><div class="ssub">Active input</div></div><div class="stat"><div class="slabel">Top-3 Accuracy</div><div class="sval">{metrics.get("top3_accuracy",0)*100:.1f}%</div><div class="ssub">From evaluation run</div></div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.markdown('<div class="wrap"><div class="ctitle">🧪 Field Analysis</div><div class="csub">Enter your soil and climate measurements below</div>', unsafe_allow_html=True)
        with st.form("crop_form"):
            f1, f2 = st.columns(2, gap="large")
            with f1:
                n = st.number_input("Nitrogen (N)", 0.0, 140.0, float(st.session_state["N"]), 1.0)
                p = st.number_input("Phosphorus (P)", 0.0, 145.0, float(st.session_state["P"]), 1.0)
                k = st.number_input("Potassium (K)", 0.0, 205.0, float(st.session_state["K"]), 1.0)
                ph = st.number_input("Soil pH", 3.5, 9.9, float(st.session_state["ph"]), 0.01, format="%.2f")
            with f2:
                temp = st.number_input("Temperature (°C)", 8.0, 44.0, float(st.session_state["temperature"]), 0.1, format="%.1f")
                hum = st.number_input("Humidity (%)", 14.0, 99.0, float(st.session_state["humidity"]), 0.1, format="%.1f")
                rain = st.number_input("Rainfall (mm)", 20.0, 300.0, float(st.session_state["rainfall"]), 0.1, format="%.1f")
            submitted = st.form_submit_button("🌱 Get Crop Recommendation", use_container_width=True, type="primary")
        st.session_state.update({"N": n, "P": p, "K": k, "ph": ph, "temperature": temp, "humidity": hum, "rainfall": rain})
        if submitted:
            render_result(model)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="wrap"><div class="ctitle">📊 Soil Nutrient Profile</div><div class="csub">Visual breakdown of your field values</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(npk_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="wrap"><div class="ctitle">🌧️ Weekly Rainfall Forecast</div><div class="csub">Detected-location precipitation outlook for the next 5 days</div>', unsafe_allow_html=True)
        st.plotly_chart(weather_chart("rain"), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="wrap"><div class="ctitle">🌡️ Temperature Trend</div><div class="csub">Daily high temperatures this week</div>', unsafe_allow_html=True)
        st.plotly_chart(weather_chart("temp"), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="wrap"><div class="ctitle">🌾 Crop Input Presets</div><div class="csub">Click any crop below to autofill representative growing conditions</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, preset in enumerate(CROP_PRESETS):
        with cols[i % 4]:
            if st.button(f'{preset["emoji"]} {preset["name"]}', key=f'preset_{preset["name"]}', use_container_width=True):
                apply_preset(preset)
                st.rerun()
            st.markdown(f'<div class="chip"><div style="font-size:28px;">{preset["emoji"]}</div><div class="chip-name">{preset["name"]}</div><div class="chip-season">{preset["season"]}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#5a7c65;font-size:12px;margin-top:1rem;">CropWise · Smart Agronomic Intelligence · Integrated with the trained recommendation pipeline</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
