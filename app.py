import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Water Tank Prediction", page_icon="💧", layout="centered")
st.title("💧 Water Tank Remaining-Water Predictor")
st.write(
    "Change the water usage rate and immediately compare how much water may remain "
    "at different usage rates. The trained Linear Regression model provides the ML prediction."
)

MODEL_PATH = Path("water_tank_model.pkl")

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: water_tank_model.pkl. Save the trained model from your Colab "
        "notebook in the same folder as app.py."
    )
    st.code('import joblib\njoblib.dump(model, "water_tank_model.pkl")', language="python")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.subheader("Tank information")
col1, col2 = st.columns(2)
with col1:
    tank_capacity = st.number_input("Tank capacity (litres)", min_value=1.0, value=1000.0, step=100.0)
    current_water = st.number_input("Current water level (litres)", min_value=0.0, value=800.0, step=10.0)
with col2:
    hours_ahead = st.slider("Predict after how many hours?", min_value=1, max_value=24, value=5)
    usage_rate = st.slider(
        "Change water usage rate (litres/hour)",
        min_value=0, max_value=200, value=40, step=5,
        help="Move this slider to see how increasing or decreasing usage changes the prediction."
    )

if current_water > tank_capacity:
    st.error("Current water level cannot exceed tank capacity.")
    st.stop()

def predict_ml(rate):
    features = pd.DataFrame([{
        "Time_Ahead_Hours": float(hours_ahead),
        "Water_Level_L": float(current_water),
        "Usage_Rate_L_per_hour": float(rate)
    }])
    value = float(np.asarray(model.predict(features)).reshape(-1)[0])
    # Keep display within plausible bounds, assuming no refill.
    return float(np.clip(value, 0, min(tank_capacity, current_water)))

# Compare machine-learning prediction with a transparent constant-usage calculation.
ml_prediction = predict_ml(usage_rate)
balance_prediction = max(0.0, current_water - usage_rate * hours_ahead)

st.subheader("Prediction at the selected usage rate")
m1, m2, m3 = st.columns(3)
m1.metric("ML predicted water", f"{ml_prediction:,.1f} L")
m2.metric("Simple estimate", f"{balance_prediction:,.1f} L")
m3.metric("Water used by simple estimate", f"{min(current_water, usage_rate * hours_ahead):,.1f} L")

st.progress(
    min(max(ml_prediction / tank_capacity, 0.0), 1.0),
    text=f"ML prediction: {ml_prediction / tank_capacity * 100:.1f}% of tank capacity"
)

if balance_prediction <= 0 and usage_rate > 0:
    st.warning("At this usage rate, the simple estimate says the tank may run empty within the selected time.")
elif balance_prediction < 0.2 * tank_capacity:
    st.warning("The simple estimate shows a low water level. Consider refilling soon.")

st.divider()
st.subheader("What happens when usage rate changes?")
st.write(
    "The chart compares the model's prediction and the simple water-balance estimate at different "
    "usage rates. The simple estimate assumes no refill and a constant usage rate."
)

rates = sorted(set([0, 20, 40, 60, 80, 100, 120, int(usage_rate)]))
rates = [r for r in rates if r <= 200]
comparison = pd.DataFrame({
    "Usage rate (L/hour)": rates,
    "ML prediction (L)": [predict_ml(r) for r in rates],
    "Simple estimate (L)": [max(0.0, current_water - r * hours_ahead) for r in rates]
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

chart_data = comparison.set_index("Usage rate (L/hour)")
st.line_chart(chart_data, y=["ML prediction (L)", "Simple estimate (L)"])

# Compare current slider value with a baseline rate of 40 L/hour.
baseline_rate = 40
baseline_ml = predict_ml(baseline_rate)
baseline_simple = max(0.0, current_water - baseline_rate * hours_ahead)
delta_ml = ml_prediction - baseline_ml
delta_simple = balance_prediction - baseline_simple

st.subheader("Effect compared with 40 L/hour")
st.write(f"Selected rate: **{usage_rate} L/hour**; baseline: **{baseline_rate} L/hour**.")
st.write(f"ML prediction difference: **{delta_ml:+.1f} L** compared with the baseline.")
st.write(f"Simple estimate difference: **{delta_simple:+.1f} L** compared with the baseline.")
if usage_rate > baseline_rate:
    st.info("Usage is higher than the baseline. The simple estimate predicts less water remaining. Check the ML line too: the model may not show the expected relationship if training data is limited or simulated.")
elif usage_rate < baseline_rate:
    st.info("Usage is lower than the baseline. The simple estimate predicts more water remaining.")
else:
    st.info("The selected rate equals the baseline rate.")

st.divider()
st.subheader("How usage rate is calculated from readings")
st.latex(r"\text{Usage rate} = \frac{\text{Previous level} - \text{Current level}}{\text{Elapsed hours}}")
st.write(
    "Example: a drop from 800 L to 760 L in one hour gives a net rate of 40 L/hour, "
    "provided no water was added during that interval."
)

st.caption(
    "Important: the ML result comes from your saved model. The simple estimate is a separate reference "
