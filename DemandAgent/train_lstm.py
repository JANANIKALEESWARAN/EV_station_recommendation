import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/demand_history.csv")

print(df.head())

# ==========================================
# Features and Target
# ==========================================

X = df[["Hour", "Weekday", "Power"]].values
y = df["Demand"].values

# ==========================================
# Normalize
# ==========================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))

# Save scalers
os.makedirs("models", exist_ok=True)

joblib.dump(x_scaler, "models/x_scaler.pkl")
joblib.dump(y_scaler, "models/y_scaler.pkl")

# ==========================================
# Reshape for LSTM
# ==========================================

X = X.reshape((X.shape[0], 1, X.shape[1]))

# ==========================================
# Build LSTM
# ==========================================

model = Sequential()

model.add(LSTM(64, input_shape=(1,3)))

model.add(Dense(32, activation="relu"))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ==========================================
# Train
# ==========================================

early = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early],
    verbose=1
)

# ==========================================
# Save Model
# ==========================================

model.save("models/demand_lstm.keras")

print("\nModel Saved Successfully!")