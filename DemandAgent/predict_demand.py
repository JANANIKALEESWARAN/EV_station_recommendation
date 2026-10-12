import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# =====================================
# Load Trained LSTM Model
# =====================================

print("Loading LSTM Model...")

model = load_model("models/demand_lstm.keras")

print("Model Loaded Successfully")

# =====================================
# Load Scalers
# =====================================

x_scaler = joblib.load("models/x_scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")

print("Scalers Loaded Successfully")

# =====================================
# User Input
# =====================================

print("\nEnter Charging Station Details\n")

hour = int(input("Hour (0-23): "))

weekday = int(input("Weekday (0=Mon ... 6=Sun): "))

power = float(input("Charging Power (kW): "))

# =====================================
# Prepare Input
# =====================================

sample = np.array([[hour, weekday, power]])

sample_scaled = x_scaler.transform(sample)

sample_scaled = sample_scaled.reshape((1, 1, 3))

# =====================================
# Predict Demand
# =====================================

prediction = model.predict(sample_scaled, verbose=0)

prediction = y_scaler.inverse_transform(prediction)

predicted_demand = prediction[0][0]

# Avoid negative prediction

if predicted_demand < 0:
    predicted_demand = 0

# =====================================
# Demand Level
# =====================================

if predicted_demand < 10:
    level = "LOW"

elif predicted_demand < 25:
    level = "MEDIUM"

else:
    level = "HIGH"

# =====================================
# Display Result
# =====================================

print("\n")
print("=" * 50)
print("        DEMAND PREDICTION RESULT")
print("=" * 50)

print(f"Hour                 : {hour}")
print(f"Weekday              : {weekday}")
print(f"Charging Power       : {power} kW")
print(f"Predicted Demand     : {predicted_demand:.2f} EVs/hour")
print(f"Demand Level         : {level}")

print("=" * 50)