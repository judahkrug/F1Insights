import fastf1
import matplotlib
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from fastf1 import plotting
from fastf1.ergast import Ergast


def enable_cache(cache_dir='cache'):
    fastf1.Cache.enable_cache(cache_dir)


def load_race(year, grand_prix, session):
    race = fastf1.get_session(year, grand_prix, session)
    race.load()
    return race


def get_fastest_lap(race, driver):
    laps = race.laps.pick_drivers(driver)
    fastest_lap = laps.pick_fastest()
    return fastest_lap


def get_average_laptimes(race, driver):
    laps = race.laps.pick_drivers(driver)
    laps = laps.dropna(subset=['LapTime'])  # Ignore laps with NaT LapTime
    # TODO: Return 0 if the driver didn't finish the race

    if len(laps) == 0:
        return 0

    total_time = sum(laps['LapTime'].dt.total_seconds())
    average_time = total_time / len(laps)
    return average_time


def get_tire_multiplier(stintLaps):
    if len(stintLaps) < 5:
        return 0  # Not enough laps to calculate multiplier

    first_lap = stintLaps[0][1]
    last_lap = stintLaps[-1][1]

    # Calculate the range for the first and last 30% of the stint
    lap_range = (last_lap - first_lap) * 0.3

    initial_laps = []
    final_laps = []

    for lap in stintLaps:
        lap_time = lap[0].total_seconds()
        if lap[1] <= first_lap + lap_range:
            initial_laps.append(lap_time)
        elif lap[1] >= last_lap - lap_range:
            final_laps.append(lap_time)

    initial_laps_avg = np.average(initial_laps)
    final_laps_avg = np.average(final_laps)

    if final_laps_avg == 0:
        return 0  # Avoid division by zero

    tire_multiplier = final_laps_avg / initial_laps_avg
    return tire_multiplier


def is_pit_lap(index, laps):
    if pd.notnull(laps.PitInTime[index]) or pd.notnull(laps.PitOutTime[index]):
        return True
    return False

def add_points_to_matrix(tire_matrix):
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=2024, round="last", result_type = "raw")
    standings = standings[0]['DriverStandings']

    driver_points = {}

    for entry in standings:
        driver_code = entry['Driver']['code']
        points = entry['points']
        driver_points[driver_code] = points

    tire_matrix['Points'] = tire_matrix.index.map(driver_points)
    print(tire_matrix.corr()['Points'])


def plot_comparison(d1, d2, d1_telemetry, d2_telemetry, team_d1, team_d2, delta_time, ref_tel):
    plot_size = [15, 15]
    plot_title = f"2020 Austrian Grand Prix - Race - {d1} VS {d2}"
    plot_ratios = [1, 3, 2, 1, 1, 2, 1]
    plot_filename = plot_title.replace(" ", "") + ".png"

    matplotlib.pyplot.rcParams['figure.figsize'] = plot_size
    fig, ax = matplotlib.pyplot.subplots(7, gridspec_kw={'height_ratios': plot_ratios})
    ax[0].title.set_text(plot_title)

    ax[0].plot(ref_tel['Distance'], delta_time)
    ax[0].axhline(0)
    ax[0].set(ylabel=f"Gap to {d2} (s)")

    ax[1].plot(d1_telemetry['Distance'], d1_telemetry['Speed'], label=d1, color=fastf1.plotting.team_color(team_d1))
    ax[1].plot(d2_telemetry['Distance'], d2_telemetry['Speed'], label=d2, color=fastf1.plotting.team_color(team_d2))
    ax[1].set(ylabel='Speed')
    ax[1].legend(loc="lower right")

    ax[2].plot(d1_telemetry['Distance'], d1_telemetry['Throttle'], label=d1, color=fastf1.plotting.team_color(team_d1))
    ax[2].plot(d2_telemetry['Distance'], d2_telemetry['Throttle'], label=d2, color=fastf1.plotting.team_color(team_d2))
    ax[2].set(ylabel='Throttle')

    ax[3].plot(d1_telemetry['Distance'], d1_telemetry['Brake'], label=d1, color=fastf1.plotting.team_color(team_d1))
    ax[3].plot(d2_telemetry['Distance'], d2_telemetry['Brake'], label=d2, color=fastf1.plotting.team_color(team_d2))
    ax[3].set(ylabel='Brake')

    ax[4].plot(d1_telemetry['Distance'], d1_telemetry['nGear'], label=d1, color=fastf1.plotting.team_color(team_d1))
    ax[4].plot(d2_telemetry['Distance'], d2_telemetry['nGear'], label=d2, color=fastf1.plotting.team_color(team_d2))
    ax[4].set(ylabel='Gear')

    ax[5].plot(d1_telemetry['Distance'], d1_telemetry['RPM'], label=d1, color=fastf1.plotting.team_color(team_d1))
    ax[5].plot(d2_telemetry['Distance'], d2_telemetry['RPM'], label=d2, color=fastf1.plotting.team_color(team_d2))
    ax[5].set(ylabel='RPM')

    ax[6].plot(d1_telemetry['Distance'], d1_telemetry['DRS'], label=d1, color=fastf1.plotting.team_color(team_d1))
    ax[6].plot(d2_telemetry['Distance'], d2_telemetry['DRS'], label=d2, color=fastf1.plotting.team_color(team_d2))
    ax[6].set(ylabel='DRS')
    ax[6].set(xlabel='Lap distance (meters)')

    for a in ax.flat:
        a.label_outer()

    matplotlib.pyplot.savefig(plot_filename, dpi=300)
    matplotlib.pyplot.show()
