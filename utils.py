import fastf1
import pandas as pd
from fastf1.ergast import Ergast

def enable_cache(cache_dir='cache'):
    fastf1.Cache.enable_cache(cache_dir)

def load_race(year, grand_prix, session):
    race = fastf1.get_session(year, grand_prix, session)
    race.load()
    return race

def get_races(year):
    events = fastf1.get_event_schedule(year)
    races = []
    for i in range(events.shape[0]):
        if events['Session5'][i] == 'Race':
            race_info = (events['OfficialEventName'][i], events['Country'][i], i)
            races.append(race_info)
    return races

def populate_missing_entries(matrix, year):
    for driver in matrix.index:
        deviations = []

        # Populate the multipliers
        for race in matrix.columns:
            driver_metric = matrix.at[driver, race]
            if pd.notna(driver_metric):
                race_average_metric = matrix[race].mean()
                deviations.append(driver_metric / race_average_metric)

        # Fill in the missing entries
        if len(deviations) == 0:
            continue
        average_deviation = sum(deviations) / len(deviations)
        for race in matrix.columns:
            if pd.isna(matrix.at[driver, race]):
                matrix.at[driver, race] = average_deviation * matrix[race].min()

    return matrix

def is_pit_lap(index, laps):
    if pd.notnull(laps.PitInTime[index]) or pd.notnull(laps.PitOutTime[index]):
        return True
    return False

def add_points_to_matrix(tire_matrix, year):
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=year, round="last", result_type="raw")
    standings = standings[0]['DriverStandings']

    driver_points = {}

    for entry in standings:
        driver_code = entry['Driver']['code']
        points = entry.get('points', 0)  # Set points to 0 if not available
        driver_points[driver_code] = points

    tire_matrix[str(year) + ' Points'] = tire_matrix['Driver'].map(driver_points).fillna(0)
    tire_matrix.to_csv('/Users/judahkrug/Desktop/F1-Data/' + str(year) + '_multipliers_with_points.csv', index=False)