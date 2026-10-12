import requests
import pandas as pd
import random
from datetime import datetime, timedelta

# ==========================================
# API KEY
# ==========================================

API_KEY = "ce2ce36f-f00d-46b0-951e-ee0b6eebf061"

# ==========================================
# LOCATION
# ==========================================

LATITUDE = 11.0168
LONGITUDE = 76.9558

url = "https://api.openchargemap.io/v3/poi"

params = {

    "output":"json",

    "countrycode":"IN",

    "latitude":LATITUDE,

    "longitude":LONGITUDE,

    "distance":30,

    "distanceunit":"KM",

    "maxresults":20,

    "key":API_KEY
}

print("Downloading charging stations...")

response = requests.get(url,params=params)

stations = response.json()

print("Stations Found :",len(stations))

rows=[]

start=datetime(2025,1,1)

days=30

for station in stations:

    station_name=station["AddressInfo"]["Title"]

    power=50

    if len(station["Connections"])>0:

        power=station["Connections"][0].get("PowerKW",50)

    for day in range(days):

        for hour in range(24):

            date=start+timedelta(days=day,hours=hour)

            weekday=date.weekday()

            demand=random.randint(2,5)

            if hour>=8 and hour<=10:

                demand=random.randint(20,35)

            elif hour>=17 and hour<=21:

                demand=random.randint(25,45)

            elif hour>=11 and hour<=16:

                demand=random.randint(10,20)

            if weekday>=5:

                demand=int(demand*0.7)

            demand+=int(power/50)

            rows.append({

                "Station":station_name,

                "Hour":hour,

                "Weekday":weekday,

                "Power":power,

                "Demand":demand

            })

df=pd.DataFrame(rows)

df.to_csv("data/demand_history.csv",index=False)

print(df.head())

print()

print("Dataset Created Successfully")

print("Rows :",len(df))