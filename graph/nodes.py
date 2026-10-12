from RecommendationAgent.recommendation_service import recommendation_agent
from DemandAgent.demand_service import demand_agent
from BatteryAgent.battery_service import battery_agent
from OptimizationAgent.optimization_service import optimize_stations
def recommendation_node(state):

    print("\nRecommendation Agent Running...")

    stations = recommendation_agent(

        battery=state["battery_percentage"],

        vehicle=state["vehicle"],

        time=state["time"],

        latitude=state["latitude"],

        longitude=state["longitude"]

    )

    state["stations"] = stations

    print(f"{len(stations)} stations received:")
    
    for i, station in enumerate(stations):
        print(f"  [{i+1}] {station.get('Station', 'Unknown')}")
        print(f"      Distance: {station.get('Distance', 'N/A')} KM")
        print(f"      Power: {station.get('Power', 'N/A')} kW")
        print(f"      Connectors: {station.get('Connectors', 'N/A')}")
        prob = station.get('Probability', 0)
        print(f"      Probability: {round(prob * 100, 2)}%\n")

    return state

def demand_node(state):

    print("\nDemand Agent Running...")

    print("Received Hour:", state["hour"])
    print("Received Weekday:", state["weekday"])
    print("Received Charging Power:", state["charging_power"])

    predicted_demand = demand_agent(

        state["hour"],
        state["weekday"],
        state["charging_power"]

    )

    state["predicted_demand"] = predicted_demand

    print(
        "Predicted Demand :",
        predicted_demand,
        "EVs/hour"
    )

    return state


def battery_node(state):

    print("\nBattery Agent Running...")

    soh = battery_agent(state.get("battery_percentage", 100.0))

    state["battery_health"] = soh

    print(f"Battery Health : {soh:.2f}%")

    return state


def optimization_node(state):

    print("\nOptimization Agent Running...")

    stations = state["stations"]

    predicted_demand = state["predicted_demand"]

    battery_health = state["battery_health"]


    optimized_station = optimize_stations(

        stations,

        predicted_demand,

        battery_health

    )


    state["optimized_station"] = optimized_station

    if optimized_station:
        print("Optimized Station:", optimized_station.get("Station", "Unknown"))
        print("Optimization Score:", optimized_station.get("OptimizationScore", "N/A"))
    else:
        print("Optimized Station: None (No stations found)")

    return state