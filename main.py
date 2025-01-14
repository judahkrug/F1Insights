import fastf1
import pandas as pd

from utils import enable_cache, load_race, get_average_laptimes


def main():
    enable_cache()
    year = 2024
    events = fastf1.get_event_schedule(year)  # Get all events in year 2024
    races = []

    # Add each Race (Session 5 of weekend) to races
    for i in range(events.shape[0]):
        if events['Session5'][i] == 'Race':
            race_info = [events['OfficialEventName'][i], events['Country'][i], i]
            races.append(race_info)

    # Get a list of unique drivers
    # TODO: Rewrite this function, it can be added into the below for race in races loop and only create a new driver row when a new driver is encountered
    unique_drivers = set()
    # for race in races:
    #     session = load_race(year, race[0], 'R')
    #     unique_drivers.update(session.drivers)
    #     print (race[])
    #     print(unique_drivers)
    #     print()

    # Comment the above out if you'd like to loop through all drivers
    unique_drivers = {'27', '14', '38', '63', '31', '2', '55', '1', '3', '77', '11', '16', '81', '50', '30', '4', '44', '10', '61', '18', '20', '22', '23', '43', '24'}

    unique_drivers = list(unique_drivers)

    # Initialize the 2D matrix
    matrix = pd.DataFrame(index=unique_drivers, columns=[race[0] for race in races])

    # Populate the matrix with average lap times
    for index, race in enumerate(races):
        session = load_race(year, race[0], 'R')

        print(f"Loading race {index + 1} / {len(races)}")
        print("Loading " + race[0])
        print()
        for driver in session.drivers:
            avg_laptime = get_average_laptimes(session, driver)
            if avg_laptime != 0:
                matrix.at[driver, race[0]] = avg_laptime


    matrix.to_csv('/Users/judahkrug/Desktop/average_lap_times.csv')

    # Populate missing values with expected % distance from the fastest driver
    for driver in unique_drivers:
        multipliers = []

        # Populate the multipliers
        for race in matrix.columns:
            driver_laptime = matrix.at[driver, race]
            if pd.notna(driver_laptime):
                fastest_laptime = matrix[race].min()
                multipliers.append(driver_laptime / fastest_laptime)

        # Fill in the missing entries with multiplier
        if len(multipliers) == 0:
            break
        multiplier = sum(multipliers) / len(multipliers)
        print("Driver: " + driver + " Multiplier: " + str(multiplier))
        for race in matrix.columns:
            if pd.isna(matrix.at[driver, race]):
                matrix.at[driver, race] = multiplier * matrix[race].min()

        


    # Print the matrix
    matrix.to_csv('/Users/judahkrug/Desktop/updated_lap_times.csv')

if __name__ == "__main__":
    main()