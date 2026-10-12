import os
import joblib
import numpy as np

from tensorflow.keras.models import load_model

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "demand_lstm.keras"
)

SCALER_X = os.path.join(
    BASE_DIR,
    "models",
    "x_scaler.pkl"
)

SCALER_Y = os.path.join(
    BASE_DIR,
    "models",
    "y_scaler.pkl"
)

# ==========================================================
# Load Model
# ==========================================================

model = load_model(MODEL_PATH)

print("Demand LSTM Loaded")

x_scaler = joblib.load(SCALER_X)

y_scaler = joblib.load(SCALER_Y)

print("Demand Scalers Loaded")

# ==========================================================
# Prediction Function
# ==========================================================

def demand_agent(hour, weekday, charging_power):

    sample = np.array([[

        hour,

        weekday,

        charging_power

    ]])

    sample = x_scaler.transform(sample)

    sample = sample.reshape((1, 1, 3))

    prediction = model.predict(sample, verbose=0)

    demand = y_scaler.inverse_transform(prediction)

    return float(demand[0][0])