from typing import TypedDict


class EVState(TypedDict):

    battery_percentage: int

    vehicle: str

    time: str

    hour: int

    weekday: int

    latitude: float

    longitude: float

    charging_power: float

    stations: list

    predicted_demand: float

    battery_health: float

    optimized_station: dict