import pandas as pd
import numpy as np

from utils import is_pit_lap, load_race

def populate_tire_multipliers(year, races, tire_matrix):
    for race in races:
        session = load_race(year, race[0], 'R')

        # Add the race to the matrix if not already present
        if race[0] not in tire_matrix.columns:
            tire_matrix[race[0]] = pd.Series(dtype='float64')

        print(f"Loading race {race[2]} / {len(races)}")
        print("Loading " + race[0])
        print()

        for driver in session.results.Abbreviation.values:
            # Add the driver to the matrix if not already present
            if driver not in tire_matrix.index:
                tire_matrix.loc[driver] = pd.Series(dtype='float64')

            tire_multipliers = []
            stint_laps = []
            laps = session.laps.pick_drivers(driver)

            # Remove drivers who completed < 75% the race
            if laps.shape[0] < .75 * session.total_laps:
                break

            # Populate tire_multipliers
            for index, lapNumber in laps.LapNumber.items():
                stint_number = laps.Stint[index]

                if lapNumber != 1 and not is_pit_lap(index, laps) and pd.notnull(laps.LapTime[index]):
                    stint_laps.append((laps.LapTime[index], lapNumber))

                if index + 1 not in laps.Stint or stint_number != laps.Stint[index + 1]:
                    tire_multiplier = get_tire_multiplier(stint_laps)
                    if tire_multiplier != 0:
                        tire_multipliers.append(tire_multiplier)
                    stint_laps.clear()

            if len(tire_multipliers) >= 1:
                avg_multiplier = np.average(tire_multipliers)
                tire_matrix.at[driver, race[0]] = avg_multiplier

    # tire_matrix.to_csv('/Users/judahkrug/Desktop/F1-Data/' + str(year) + '_tire_multipliers.csv')
    return tire_matrix


def get_tire_multiplier(stint_laps):
    if len(stint_laps) < 5:
        return 0  # Not enough laps to calculate multiplier

    first_lap = stint_laps[0][1]
    last_lap = stint_laps[-1][1]

    # Calculate the range for the first and last 30% of the stint
    lap_range = (last_lap - first_lap) * 0.3

    initial_laps = []
    final_laps = []

    for lap in stint_laps:
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