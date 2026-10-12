import pandas as pd
import numpy as np

# -----------------------------
# Load station dataset
# -----------------------------

stations = pd.read_csv("data/stations.csv")

# Remove stations with missing distance
stations = stations.dropna(subset=["Distance"])

dataset = []

np.random.seed(42)

NUM_USERS = 10000

vehicle_types = [
    "Car",
    "Bike",
    "SUV"
]

times = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
]

for user in range(NUM_USERS):

    battery = np.random.randint(5,101)

    vehicle = np.random.choice(vehicle_types)

    time = np.random.choice(times)

    for _,row in stations.iterrows():

        distance = row["Distance"]

        power = row["PowerKW"]

        connectors = row["Connectors"]

        operator = row["Station_Name"]

        if pd.isna(power):
            power = 20

        if pd.isna(connectors):
            connectors = 1

        # ---------------------------------
        # Intelligent Label Creation
        # ---------------------------------

        score = 0

        # nearer station
        score += max(0,10-distance)

        # fast charging
        score += power/20

        # connector availability
        score += connectors

        # battery priority

        if battery < 20:
            score += 6

        elif battery < 40:
            score += 4

        elif battery < 60:
            score += 2

        # random behaviour

        score += np.random.uniform(-1,1)

        selected = 1 if score >= 12 else 0

        dataset.append({

            "Battery":battery,

            "Distance":distance,

            "Power":power,

            "Connectors":connectors,

            "Vehicle":vehicle,

            "Time":time,

            "Selected":selected

        })

train = pd.DataFrame(dataset)

train.to_csv("data/train.csv",index=False)

print(train.head())

print()

print("Total Samples :",len(train))