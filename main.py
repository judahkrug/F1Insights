import fastf1
import pandas as pd

from populate_tire_multipliers import populate_tire_multipliers
from utils import enable_cache, add_points_to_matrix, get_races, populate_missing_entries


def main():
    enable_cache()
    year = 2023
    events = fastf1.get_event_schedule(year)
    races = get_races(year)
    tire_matrix = pd.DataFrame()


    # Populate matrix with Tire Multipliers
    tire_matrix = populate_tire_multipliers(year, races, tire_matrix)

    # Populate missing entries with average deviation from race average tire_multipliers
    tire_matrix = populate_missing_entries(tire_matrix, year)

    # tire_matrix = pd.read_csv('/Users/judahkrug/Desktop/2023_tire_multipliers.csv', index_col=0)
    add_points_to_matrix(tire_matrix, year)


    print("Correlations Below:")
    print(tire_matrix.corr()['Points'])
    print("Execution Done")


if __name__ == "__main__":
    main()
