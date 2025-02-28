import fastf1
from fastf1 import utils
from src.utils.helpers import enable_cache, load_race, get_fastest_lap, plot_comparison


def compare_fastest():
    enable_cache()
    race = load_race(2020, 'Austrian Grand Prix', 'R')

    d1, d2 = 'NOR', 'HAM'
    team_d1, team_d2 = 'MCLAREN', 'MERCEDES'

    # Fastest Lap Comparison
    d1_fastest_lap = get_fastest_lap(race, d1)
    d2_fastest_lap = get_fastest_lap(race, d2)

    d1_telemetry = d1_fastest_lap.get_telemetry().add_distance()
    d2_telemetry = d2_fastest_lap.get_telemetry().add_distance()

    delta_time, ref_tel, compare_tel = fastf1.utils.delta_time(d1_fastest_lap, d2_fastest_lap)

    plot_comparison(d1, d2, d1_telemetry, d2_telemetry, team_d1, team_d2, delta_time, ref_tel)


def compare_last():
    enable_cache()
    race = load_race(2020, 'Austrian Grand Prix', 'R')

    d1, d2 = 'NOR', 'HAM'
    team_d1, team_d2 = 'MCLAREN', 'MERCEDES'

    # Last Lap Comparison
    d1_last_lap = get_last_lap(race, d1)
    d2_last_lap = get_last_lap(race, d2)

    d1_telemetry = d1_last_lap.get_telemetry().add_distance()
    d2_telemetry = d2_last_lap.get_telemetry().add_distance()

    delta_time, ref_tel, compare_tel = fastf1.utils.delta_time(d1_last_lap, d2_last_lap)

    plot_comparison(d1, d2, d1_telemetry, d2_telemetry, team_d1, team_d2, delta_time, ref_tel)


def print_lap_times():
    enable_cache()
    session = load_race(2020, 'Austrian Grand Prix', 'R')
    columns = ["Driver", "Position", "LapNumber", "LapTime", "Compound", "TyreLife"]
    laps = session.laps[columns].to_numpy()

    # Print column headers
    print(columns)

    # Print the 2D array
    for lap in laps:
        print(lap)

    print("Done!")
