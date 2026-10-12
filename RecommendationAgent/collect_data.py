import requests
import pandas as pd

# ==============================
# API KEY
# ==============================

API_KEY = "ce2ce36f-f00d-46b0-951e-ee0b6eebf061"

# ==============================
# USER LOCATION
# ==============================

LATITUDE = 11.0168       # Coimbatore
LONGITUDE = 76.9558

# ==============================
# API URL
# ==============================

url = "https://api.openchargemap.io/v3/poi"

params = {

    "output": "json",

    "countrycode": "IN",

    "latitude": LATITUDE,

    "longitude": LONGITUDE,

    "distance": 100,

    "distanceunit": "KM",

    "maxresults": 100

}

headers = {

    "X-API-Key": API_KEY

}

print("Downloading charging stations...")

response = requests.get(url, params=params, headers=headers)

if response.status_code != 200:

    print("API Error")

    print(response.text)

    exit()

stations = response.json()

print("Stations Found :", len(stations))

# ==============================
# CREATE DATASET
# ==============================

rows = []

for station in stations:

    address = station.get("AddressInfo", {})

    connections = station.get("Connections", [])

    power = 0
    connectors = 0

    if len(connections) > 0:

        power = connections[0].get("PowerKW",0)

        connectors = connections[0].get("Quantity",1)

    rows.append({

        "Station_Name" : address.get("Title"),

        "Latitude" : address.get("Latitude"),

        "Longitude" : address.get("Longitude"),

        "Distance" : address.get("Distance"),

        "PowerKW" : power,

        "Connectors" : connectors,

        "Address" : address.get("AddressLine1"),

        "Town" : address.get("Town"),

        "State" : address.get("StateOrProvince")

    })

# ==============================
# SAVE CSV
# ==============================

df = pd.DataFrame(rows)

df.to_csv("data/stations.csv",index=False)

print()

print(df.head())

print()

print("Dataset Saved Successfully!")