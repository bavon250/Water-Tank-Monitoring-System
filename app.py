
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Water Tank Monitoring",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Water Tank Monitoring System")
st.subheader("Predict Remaining Water Using Linear Regression")


# Load the existing project dataset automatically
DATA_FILE = (
    Path(__file__).parent
    / "water_tank_dataset_time_level_usage.csv"
)

@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    numeric_columns = [
        "water_level_percent",
        "usage_rate_liters_per_min",
        "remaining_water_liters"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["time"] + numeric_columns
    df = df.dropna(subset=required)
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError("The dataset must contain at least 5 valid rows.")

    return df

try:
    if not DATA_FILE.exists():
        st.error(
            "Dataset not found. Make sure "
            "'water_tank_dataset_time_level_usage.csv' "
            "is in the same GitHub folder as app.py."
        )
        st.stop()

    df = load_dataset(str(DATA_FILE))

except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

# Prepare data for machine learning
df["elapsed_minutes"] = (
    df["time"] - df["time"].min()
).dt.total_seconds() / 60

features = [
    "elapsed_minutes",
    "water_level_percent",
    "usage_rate_liters_per_min"
]

target = "remaining_water_liters"

X = df[features]
y = df[target]

# Train on earlier readings and test on later readings
split = int(len(df) * 0.8)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model automatically
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Dashboard
st.success("Model trained successfully using the included dataset.")

st.header("1. Tank overview")

a, b, c = st.columns(3)
a.metric(
    "Latest water level",
    f"{df['water_level_percent'].iloc[-1]:.1f}%"
)
b.metric(
    "Latest remaining water",
    f"{df['remaining_water_liters'].iloc[-1]:.1f} L"
)
c.metric(
    "Latest usage rate",
    f"{df['usage_rate_liters_per_min'].iloc[-1]:.2f} L/min"
)

with st.expander("View the training dataset"):
    st.dataframe(
        df[[
            "time",
            "water_level_percent",
            "usage_rate_liters_per_min",
            "remaining_water_liters"
        ]],
        use_container_width=True
    )

# Model evaluation
st.header("2. Model performance")

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f} liters")
m2.metric("RMSE", f"{rmse:.2f} liters")
m3.metric(
    "R² score",
    f"{r2:.4f}" if np.isfinite(r2) else "N/A"
)

st.caption(
    "MAE and RMSE measure prediction error in liters. "
    "Lower values generally indicate smaller errors."
)

# Actual versus predicted graph
st.subheader("Actual vs predicted water")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    df["time"].iloc[split:],
    y_test.to_numpy(),
    label="Actual water"
)
ax.plot(
    df["time"].iloc[split:],
    predictions,
    label="Predicted water"
)
ax.set_xlabel("Time")
ax.set_ylabel("Remaining water (liters)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# User prediction form
st.header("3. Predict remaining water")
st.write("Enter the current tank readings below.")

col1, col2 = st.columns(2)

with col1:
    water_level = st.number_input(
        "Current water level (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    usage_rate = st.number_input(
        "Current usage rate (liters/minute)",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    elapsed_minutes = st.number_input(
        "Minutes since the first dataset reading",
        min_value=0.0,
        value=float(df["elapsed_minutes"].iloc[-1]),
        step=5.0
    )

    tank_capacity = st.number_input(
        "Tank capacity (liters)",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

    predict_button = st.button(
        "Predict Remaining Water",
        type="primary",
        use_container_width=True
    )

with col2:
    st.info(
        "Enter the water level and usage rate from your tank. "
        "The time value should use the same reference as the training data."
    )

if predict_button:
    new_reading = pd.DataFrame([{
        "elapsed_minutes": elapsed_minutes,
        "water_level_percent": water_level,
        "usage_rate_liters_per_min": usage_rate
    }])

    raw_prediction = float(model.predict(new_reading[features])[0])

    # Limit the displayed estimate to the possible tank capacity
    estimated_water = float(
        np.clip(raw_prediction, 0, tank_capacity)
    )

    st.divider()
    st.subheader("Prediction result")

    r1, r2 = st.columns(2)
    r1.metric(
        "Estimated remaining water",
        f"{estimated_water:,.2f} liters"
    )
    r2.metric(
        "Estimated tank level",
        f"{estimated_water / tank_capacity * 100:.1f}%"
    )

    if raw_prediction < 0 or raw_prediction > tank_capacity:
        st.warning(
            "The raw model prediction was outside the tank capacity "
            "range. The displayed estimate has been limited to that range."
        )

    st.progress(
        min(1.0, max(0.0, estimated_water / tank_capacity))
    )

# Water level history
st.header("4. Water level history")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["time"], df["water_level_percent"])
ax2.set_xlabel("Time")
ax2.set_ylabel("Water level (%)")
ax2.set_ylim(0, 100)
fig2.autofmt_xdate()
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.warning(
    "Important: this project uses synthetic sample data. "
    "Its remaining-water values are calculated directly from water level, "
    "so evaluation results may look unusually good. The app does not "
    "connect to a real tank sensor. Validate predictions with real "
    "measurements before relying on them."
)
