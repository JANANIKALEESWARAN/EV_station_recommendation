import requests
import pandas as pd
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Load Trained Model
# ==========================================

model = TabNetClassifier()

model.load_model("models/tabnet_model.zip")

print("TabNet Model Loaded Successfully")

# ==========================================
# User Input
# ==========================================

battery = int(input("Enter Battery Percentage (0-100): "))

vehicle = input("Vehicle (Car/Bike/SUV): ").strip().title()

time = input("Time (Morning/Afternoon/Evening/Night): ").strip().title()

# ==========================================
# Encode Categories
# ==========================================

vehicle_encoder = LabelEncoder()
vehicle_encoder.fit(["Car","Bike","SUV"])

time_encoder = LabelEncoder()
time_encoder.fit([
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
])

vehicle = vehicle_encoder.transform([vehicle])[0]
time = time_encoder.transform([time])[0]

# ==========================================
# API
# ==========================================

API_KEY = "ce2ce36f-f00d-46b0-951e-ee0b6eebf061"

LATITUDE = 11.0168
LONGITUDE = 76.9558

url = "https://api.openchargemap.io/v3/poi"

params = {

    "output":"json",

    "countrycode":"IN",

    "latitude":LATITUDE,

    "longitude":LONGITUDE,

    "distance":20,

    "distanceunit":"KM",

    "maxresults":20

}

headers = {

    "X-API-Key":API_KEY

}

response = requests.get(url,params=params,headers=headers)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)

if response.status_code != 200:
    print("API Request Failed!")
    exit()

stations = response.json()

# ==========================================
# Prepare Features
# ==========================================

rows = []

display = []

for station in stations:

    address = station.get("AddressInfo",{})

    connections = station.get("Connections",[])

    if len(connections)>0:

        power = connections[0].get("PowerKW",20)

        connectors = connections[0].get("Quantity",1)

    else:

        power = 20

        connectors = 1

    distance = address.get("Distance",20)

    rows.append([

        battery,

        distance,

        power,

        connectors,

        vehicle,

        time

    ])

    display.append({

        "Station":address.get("Title"),

        "Distance":distance,

        "Power":power,

        "Connectors":connectors,

        "Latitude":address.get("Latitude"),

        "Longitude":address.get("Longitude")

    })

# ==========================================
# Predict
# ==========================================

X = np.array(rows)

prob = model.predict_proba(X)

for i in range(len(display)):

    display[i]["Probability"] = prob[i][1]

# ==========================================
# Sort
# ==========================================

display = sorted(

    display,

    key=lambda x:x["Probability"],

    reverse=True

)

# ==========================================
# Top 3
# ==========================================

print("\n")

print("="*70)

print("TOP 3 RECOMMENDED STATIONS")

print("="*70)

for station in display[:3]:

    print()

    print("Station :",station["Station"])

    print("Distance :",round(station["Distance"],2),"KM")

    print("Power :",station["Power"],"kW")

    print("Connectors :",station["Connectors"])

    print("Recommendation Score :",round(station["Probability"]*100,2),"%")

    print("Latitude :",station["Latitude"])

    print("Longitude :",station["Longitude"])

    print("-"*70)