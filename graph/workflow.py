import os
import sys

# Add project root to sys.path so imports work when running from this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from langgraph.graph import StateGraph, END
import requests

from graph.state import EVState

from graph.nodes import (
    recommendation_node,
    demand_node,
    battery_node,
    optimization_node
)


# ==========================================================
# USER INPUT
# ==========================================================

print("\n")
print("=" * 60)
print("          SMART EV MULTI-AGENT SYSTEM")
print("=" * 60)

print("\nEnter your EV details\n")


battery_percentage = float(
    input("Battery Percentage (0-100): ")
)


print("\nFetching your live location...")
try:
    loc_response = requests.get("http://ip-api.com/json/").json()
    latitude = float(loc_response['lat'])
    longitude = float(loc_response['lon'])
    print(f"Location found: {loc_response.get('city', 'Unknown City')} (Lat: {latitude}, Lon: {longitude})")
except Exception as e:
    print("Could not fetch live location. Please enter manually.")
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))


vehicle = input(
    "Vehicle Type (Car/Bike/SUV): "
).strip().title()


time = input(
    "Time (Morning/Afternoon/Evening/Night): "
).strip().title()


hour = int(
    input("Current Hour (0-23): ")
)


weekday = int(
    input("Weekday (0=Monday, 6=Sunday): ")
)


charging_power = float(
    input("Charging Power (kW): ")
)


# ==========================================================
# CREATE LANGGRAPH WORKFLOW
# ==========================================================

workflow = StateGraph(EVState)


# ==========================================================
# ADD AGENTS
# ==========================================================

workflow.add_node(

    "Recommendation",

    recommendation_node

)


workflow.add_node(

    "Demand",

    demand_node

)


workflow.add_node(

    "Battery",

    battery_node

)


workflow.add_node(

    "Optimization",

    optimization_node

)


# ==========================================================
# SET WORKFLOW START
# ==========================================================

workflow.set_entry_point(

    "Recommendation"

)


# ==========================================================
# DEFINE AGENT FLOW
# ==========================================================

workflow.add_edge(

    "Recommendation",

    "Demand"

)


workflow.add_edge(

    "Demand",

    "Battery"

)


workflow.add_edge(

    "Battery",

    "Optimization"

)


workflow.add_edge(

    "Optimization",

    END

)


# ==========================================================
# COMPILE GRAPH
# ==========================================================

graph = workflow.compile()


# ==========================================================
# INITIAL STATE
# ==========================================================

state = {


    # --------------------------------------
    # USER INPUT
    # --------------------------------------

    "battery_percentage":

        battery_percentage,


    "latitude":

        latitude,


    "longitude":

        longitude,


    "vehicle":

        vehicle,


    "time":

        time,


    "hour":

        hour,


    "weekday":

        weekday,


    "charging_power":

        charging_power,


    # --------------------------------------
    # AGENT OUTPUTS
    # --------------------------------------

    "stations": [],


    "predicted_demand":

        0.0,


    "battery_health":

        0.0,


    "optimized_station": {}

}


# ==========================================================
# RUN MULTI-AGENT WORKFLOW
# ==========================================================

print("\n")
print("=" * 60)
print("          STARTING MULTI-AGENT WORKFLOW")
print("=" * 60)


result = graph.invoke(

    state

)


# ==========================================================
# FINAL RESULT
# ==========================================================

print("\n")
print("=" * 60)
print("             FINAL RECOMMENDATION")
print("=" * 60)


optimized_station = result.get(

    "optimized_station",

    {}

)


print("\nRecommended Charging Station")


print(

    "Station :",

    optimized_station.get(

        "Station",

        "Not Available"

    )

)


print(

    "Distance :",

    optimized_station.get(

        "Distance",

        "N/A"

    ),

    "KM"

)


print(

    "Charging Power :",

    optimized_station.get(

        "Power",

        "N/A"

    ),

    "kW"

)


print(

    "Connectors :",

    optimized_station.get(

        "Connectors",

        "N/A"

    )

)


print(

    "Recommendation Probability :",

    round(

        optimized_station.get(

            "Probability",

            0

        ) * 100,

        2

    ),

    "%"

)


print(

    "Predicted Demand :",

    optimized_station.get(

        "PredictedDemand",

        result.get(

            "predicted_demand",

            "N/A"

        )

    ),

    "EVs/hour"

)


print(

    "Battery Health :",

    optimized_station.get(

        "BatteryHealth",

        result.get(

            "battery_health",

            "N/A"

        )

),

    "%"

)


print(

    "Optimization Score :",

    optimized_station.get(

        "OptimizationScore",

        "N/A"

    )

)


print("\n")
print("=" * 60)
print("       RECOMMENDATION COMPLETED")
print("=" * 60)