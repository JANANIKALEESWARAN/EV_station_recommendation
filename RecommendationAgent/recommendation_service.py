import requests
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Load TabNet Model (Loads only once)
# ==========================================
import os
model = TabNetClassifier()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tabnet_model.zip"
)

model.load_model(MODEL_PATH)

print("Recommendation Model Loaded")

# ==========================================
# Label Encoders
# ==========================================

vehicle_encoder = LabelEncoder()
vehicle_encoder.fit(["Car", "Bike", "SUV"])

time_encoder = LabelEncoder()
time_encoder.fit([
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
])

# ==========================================
# OpenChargeMap API
# ==========================================

import os
API_KEY = os.getenv("OPENCHARGEMAP_API_KEY", "ce2ce36f-f00d-46b0-951e-ee0b6eebf061")


# ==========================================
# Recommendation Function
# ==========================================

def recommendation_agent(battery, vehicle, time, latitude, longitude):

    # -----------------------------
    # Encode User Inputs
    # -----------------------------

    try:
        vehicle = vehicle_encoder.transform([vehicle])[0]
    except Exception:
        vehicle = vehicle_encoder.transform(["Car"])[0]

    try:
        time = time_encoder.transform([time])[0]
    except Exception:
        time = time_encoder.transform(["Morning"])[0]

    # -----------------------------
    # Call API
    # -----------------------------

    url = "https://api.openchargemap.io/v3/poi"

    params = {

        "output": "json",

        "countrycode": "IN",

        "latitude": latitude,

        "longitude": longitude,

        "distance": 20,

        "distanceunit": "KM",

        "maxresults": 20

    }

    headers = {

        "X-API-Key": API_KEY

    }

    response = requests.get(
        url,
        params=params,
        headers=headers
    )

    if response.status_code != 200:

        print("OpenChargeMap API Failed")

        return []

    try:
        stations = response.json()
    except Exception as e:
        print("OpenChargeMap API JSONDecodeError:", str(e), "- Response Text:", response.text[:200])
        return []

    if not stations:
        return []

    # -----------------------------
    # Feature Creation
    # -----------------------------

    rows = []

    display = []

    for station in stations:

        address = station.get("AddressInfo", {})

        connections = station.get("Connections", [])

        if len(connections) > 0:
            power = connections[0].get("PowerKW")
            if power is None: power = 20.0
            
            connectors = connections[0].get("Quantity")
            if connectors is None: connectors = 1
        else:
            power = 20.0
            connectors = 1

        distance = address.get("Distance")
        if distance is None: distance = 20.0

        rows.append([
            float(battery),
            float(distance),
            float(power),
            float(connectors),
            float(vehicle),
            float(time)
        ])

        display.append({

            "Station": address.get("Title"),

            "Distance": distance,

            "Power": power,

            "Connectors": connectors,

            "Latitude": address.get("Latitude"),

            "Longitude": address.get("Longitude")

        })

    # -----------------------------
    # Prediction
    # -----------------------------

    X = np.array(rows, dtype=np.float32)

    probability = model.predict_proba(X)

    for i in range(len(display)):

        display[i]["Probability"] = float(probability[i][1])

    # -----------------------------
    # Sort
    # -----------------------------

    display = sorted(

        display,

        key=lambda x: x["Probability"],

        reverse=True

    )

    # -----------------------------
    # Return Top 3 Stations
    # -----------------------------

    return display[:3]