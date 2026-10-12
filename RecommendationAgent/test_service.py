from recommendation_service import recommendation_agent

stations = recommendation_agent(

    battery=25,

    vehicle="Car",

    time="Evening"

)

for station in stations:

    print(station)