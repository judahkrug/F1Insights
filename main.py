import fastf1
import numpy as np
import pandas as pd

from utils import enable_cache, load_race, get_tire_multiplier, is_pit_lap


def main():
    enable_cache()
    year = 2024
    events = fastf1.get_event_schedule(year)
    races = []
    tire_matrix = pd.DataFrame()

    # Add each race (Session 5 of weekend) to races
    for i in range(events.shape[0]):
        if events['Session5'][i] == 'Race':
            race_info = (events['OfficialEventName'][i], events['Country'][i], i)
            races.append(race_info)

    # Populate matrix with Tire Multipliers
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

    tire_matrix.to_csv('/Users/judahkrug/Desktop/tire_multipliers.csv')

    # # Populate missing values with expected % distance from the fastest driver
    # for driver in drivers:
    #     multipliers = []
    #
    #     # Populate the multipliers
    #     for race in matrix.columns:
    #         driver_laptime = matrix.at[driver, race]
    #         if pd.notna(driver_laptime):
    #             fastest_laptime = matrix[race].min()
    #             multipliers.append(driver_laptime / fastest_laptime)
    #
    #     # Fill in the missing entries with multiplier
    #     if len(multipliers) == 0:
    #         break
    #     multiplier = sum(multipliers) / len(multipliers)
    #     print("Driver: " + driver + " Multiplier: " + str(multiplier))
    #     for race in matrix.columns:
    #         if pd.isna(matrix.at[driver, race]):
    #             matrix.at[driver, race] = multiplier * matrix[race].min()

    # # Print the matrix
    # matrix.to_csv('/Users/judahkrug/Desktop/updated_lap_times.csv')


if __name__ == "__main__":
    main()
