import fastf1
import pandas as pd

from populate_tire_multipliers import populate_tire_multipliers
from utils import enable_cache, add_points_to_matrix, get_races, populate_missing_entries

def collect_data(years):
    tire_matrix = pd.DataFrame()
    for year in years:
        races = get_races(year)
        yearly_tire_matrix = pd.DataFrame()

        # Populate matrix with Tire Multipliers
        yearly_tire_matrix = populate_tire_multipliers(year, races, yearly_tire_matrix)

        # Populate missing entries with average deviation from race average tire_multipliers
        yearly_tire_matrix = populate_missing_entries(yearly_tire_matrix, year)

        # Add points to the matrix
        add_points_to_matrix(yearly_tire_matrix, year)

        # Append to the main tire matrix
        tire_matrix = tire_matrix.combine_first(yearly_tire_matrix)
        # tire_matrix = pd.concat([tire_matrix, yearly_tire_matrix], ignore_index=False)

    return tire_matrix

def main():
    enable_cache()
    years = [2023, 2024]
    tire_matrix = collect_data(years)

    # Save the aggregated data to CSV
    tire_matrix.to_csv('/Users/judahkrug/Desktop/F1-Data/aggregated_tire_multipliers.csv')

    print("Correlations Below:")
    print(tire_matrix.corr()['2024 Points'])
    print("Execution Done")


if __name__ == "__main__":
    main()
