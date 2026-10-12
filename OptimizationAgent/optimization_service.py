import random

from deap import base
from deap import creator
from deap import tools


# ==========================================================
# CREATE DEAP FITNESS CLASSES
# ==========================================================

# Avoid creating the same classes multiple times
# when LangGraph reloads/imports the module.

if not hasattr(creator, "FitnessMulti"):

    creator.create(
        "FitnessMulti",
        base.Fitness,

        # Objectives:
        #
        # 1. Recommendation Probability -> MAXIMIZE
        # 2. Distance                 -> MINIMIZE
        # 3. Demand                   -> MINIMIZE
        # 4. Charging Power           -> MAXIMIZE
        # 5. Battery Health           -> MAXIMIZE

        weights=(

            1.0,

            -1.0,

            -1.0,

            1.0,

            1.0

        )

    )


if not hasattr(creator, "Individual"):

    creator.create(

        "Individual",

        list,

        fitness=creator.FitnessMulti

    )


# ==========================================================
# OPTIMIZATION AGENT
# ==========================================================

def optimize_stations(

        stations,

        predicted_demand,

        battery_health

):


    # ======================================================
    # CHECK STATIONS
    # ======================================================

    if stations is None or len(stations) == 0:

        print("No stations available")

        return None


    # ======================================================
    # OBJECTIVE FUNCTION
    # ======================================================

    def evaluate(individual):


        # ----------------------------------------------
        # Get Station Index
        # ----------------------------------------------

        index = int(

            individual[0]

        )


        # ----------------------------------------------
        # Keep Index Within Valid Range
        # ----------------------------------------------

        index = max(

            0,

            min(

                index,

                len(stations) - 1

            )

        )


        station = stations[index]


        # ----------------------------------------------
        # Station Features
        # ----------------------------------------------

        probability = float(

            station.get(

                "Probability",

                0

            )

        )


        distance = float(

            station.get(

                "Distance",

                20

            )

        )


        power = float(

            station.get(

                "Power",

                20

            )

        )


        # ----------------------------------------------
        # Multi-objective Values
        # ----------------------------------------------

        return (

            probability,

            distance,

            predicted_demand,

            power,

            battery_health

        )


    # ======================================================
    # TOOLBOX
    # ======================================================

    toolbox = base.Toolbox()


    # ======================================================
    # CREATE RANDOM STATION INDEX
    # ======================================================

    toolbox.register(

        "station_index",

        random.randrange,

        len(stations)

    )


    # ======================================================
    # CREATE INDIVIDUAL
    # ======================================================

    toolbox.register(

        "individual",

        tools.initRepeat,

        creator.Individual,

        toolbox.station_index,

        n=1

    )


    # ======================================================
    # CREATE POPULATION
    # ======================================================

    toolbox.register(

        "population",

        tools.initRepeat,

        list,

        toolbox.individual

    )


    # ======================================================
    # REGISTER FUNCTIONS
    # ======================================================

    toolbox.register(

        "evaluate",

        evaluate

    )


    toolbox.register(

        "mate",

        tools.cxUniform,

        indpb=0.5

    )


    toolbox.register(

        "mutate",

        tools.mutUniformInt,

        low=0,

        up=len(stations) - 1,

        indpb=0.2

    )


    toolbox.register(

        "select",

        tools.selNSGA2

    )


    # ======================================================
    # CREATE INITIAL POPULATION
    # ======================================================

    # 48 is used because:
    #
    # 48 % 4 = 0
    #
    # selTournamentDCD requires the population size
    # to be divisible by 4.

    population = toolbox.population(

        n=48

    )


    # ======================================================
    # INITIAL FITNESS EVALUATION
    # ======================================================

    for individual in population:


        individual.fitness.values = (

            toolbox.evaluate(

                individual

            )

        )


    # ======================================================
    # INITIAL NSGA-II SELECTION
    # ======================================================

    # VERY IMPORTANT
    #
    # This assigns:
    #
    # 1. Pareto rank
    # 2. Crowding distance
    #
    # selTournamentDCD requires crowding_dist.

    population = toolbox.select(

        population,

        len(population)

    )


    # ======================================================
    # EVOLUTION
    # ======================================================

    for generation in range(20):


        # --------------------------------------------------
        # TOURNAMENT SELECTION
        # --------------------------------------------------

        offspring = tools.selTournamentDCD(

            population,

            len(population)

        )


        # --------------------------------------------------
        # CLONE OFFSPRING
        # --------------------------------------------------

        offspring = list(

            map(

                toolbox.clone,

                offspring

            )

        )


        # --------------------------------------------------
        # CROSSOVER
        # --------------------------------------------------

        for child1, child2 in zip(

            offspring[::2],

            offspring[1::2]

        ):


            if random.random() < 0.7:


                toolbox.mate(

                    child1,

                    child2

                )


                # --------------------------------------
                # Keep Station Index Valid
                # --------------------------------------

                child1[0] = max(

                    0,

                    min(

                        int(child1[0]),

                        len(stations) - 1

                    )

                )


                child2[0] = max(

                    0,

                    min(

                        int(child2[0]),

                        len(stations) - 1

                    )

                )


                # --------------------------------------
                # Remove Old Fitness
                # --------------------------------------

                del child1.fitness.values

                del child2.fitness.values


        # --------------------------------------------------
        # MUTATION
        # --------------------------------------------------

        for mutant in offspring:


            if random.random() < 0.2:


                toolbox.mutate(

                    mutant

                )


                # --------------------------------------
                # Keep Station Index Valid
                # --------------------------------------

                mutant[0] = max(

                    0,

                    min(

                        int(mutant[0]),

                        len(stations) - 1

                    )

                )


                # --------------------------------------
                # Remove Old Fitness
                # --------------------------------------

                del mutant.fitness.values


        # --------------------------------------------------
        # EVALUATE INVALID INDIVIDUALS
        # --------------------------------------------------

        invalid_individuals = [

            individual

            for individual in offspring

            if not individual.fitness.valid

        ]


        for individual in invalid_individuals:


            individual.fitness.values = (

                toolbox.evaluate(

                    individual

                )

            )


        # --------------------------------------------------
        # NSGA-II SURVIVOR SELECTION
        # --------------------------------------------------

        population = toolbox.select(

            population + offspring,

            48

        )


    # ======================================================
    # GET PARETO FRONT
    # ======================================================

    pareto_front = tools.sortNondominated(

        population,

        len(population),

        first_front_only=True

    )[0]


    # ======================================================
    # COLLECT STATION VALUES
    # ======================================================

    distances = [

        float(

            station.get(

                "Distance",

                20

            )

        )

        for station in stations

    ]


    powers = [

        float(

            station.get(

                "Power",

                20

            )

        )

        for station in stations

    ]


    probabilities = [

        float(

            station.get(

                "Probability",

                0

            )

        )

        for station in stations

    ]


    # ======================================================
    # MIN/MAX VALUES
    # ======================================================

    min_distance = min(

        distances

    )


    max_distance = max(

        distances

    )


    min_power = min(

        powers

    )


    max_power = max(

        powers

    )


    min_probability = min(

        probabilities

    )


    max_probability = max(

        probabilities

    )


    # ======================================================
    # SCORE PARETO SOLUTIONS
    # ======================================================

    scored_stations = []


    for individual in pareto_front:


        # ----------------------------------------------
        # Get Station
        # ----------------------------------------------

        index = int(

            individual[0]

        )


        station = stations[index].copy()


        # ----------------------------------------------
        # Get Features
        # ----------------------------------------------

        distance = float(

            station.get(

                "Distance",

                20

            )

        )


        power = float(

            station.get(

                "Power",

                20

            )

        )


        probability = float(

            station.get(

                "Probability",

                0

            )

        )


        # ==================================================
        # DISTANCE SCORE
        # ==================================================

        if max_distance != min_distance:


            distance_score = (

                max_distance - distance

            ) / (

                max_distance - min_distance

            )


        else:


            distance_score = 1.0


        # ==================================================
        # POWER SCORE
        # ==================================================

        if max_power != min_power:


            power_score = (

                power - min_power

            ) / (

                max_power - min_power

            )


        else:


            power_score = 1.0


        # ==================================================
        # RECOMMENDATION SCORE
        # ==================================================

        if max_probability != min_probability:


            probability_score = (

                probability - min_probability

            ) / (

                max_probability - min_probability

            )


        else:


            probability_score = 1.0


        # ==================================================
        # DEMAND SCORE
        # ==================================================

        # Lower demand is better.

        demand_score = 1 / (

            1 + predicted_demand

        )


        # ==================================================
        # BATTERY HEALTH SCORE
        # ==================================================

        battery_score = (

            battery_health / 100

        )


        # ==================================================
        # FINAL OPTIMIZATION SCORE
        # ==================================================

        final_score = (

            0.30 * probability_score

            +

            0.25 * distance_score

            +

            0.20 * power_score

            +

            0.15 * demand_score

            +

            0.10 * battery_score

        )


        # ==================================================
        # ADD RESULTS
        # ==================================================

        station[

            "OptimizationScore"

        ] = round(

            final_score,

            4

        )


        station[

            "PredictedDemand"

        ] = round(

            predicted_demand,

            2

        )


        station[

            "BatteryHealth"

        ] = round(

            battery_health,

            2

        )


        scored_stations.append(

            station

        )


    # ======================================================
    # SORT STATIONS
    # ======================================================

    scored_stations = sorted(

        scored_stations,

        key=lambda station:

        station["OptimizationScore"],

        reverse=True

    )


    # ======================================================
    # RETURN BEST STATION
    # ======================================================

    return scored_stations[0]