import pandas as pd

from populate_tire_matrix import populate_tire_matrix
from utils import enable_cache, get_races


def collect_data(years):
    tire_matrix = pd.DataFrame(columns=[
        'Driver', 'Race', 'Stint', 'StintLapNumber', 'LapTime', 'Compound',
        'BaselineTime', 'DegradationPct', 'SmoothedDeg', 'PositionsGained'
    ])

    for year in years:
        races = get_races(year)
        yearly_tire_matrix = pd.DataFrame(columns=tire_matrix.columns)

        # Populate Tire Matrix
        yearly_tire_matrix = populate_tire_matrix(year, races, yearly_tire_matrix)

        # Append to the main tire matrix
        tire_matrix = pd.concat([tire_matrix, yearly_tire_matrix], ignore_index=True)

    return tire_matrix


def main():
    enable_cache()
    years = [2023, 2024]
    tire_matrix = collect_data(years)

    # Save the aggregated data to CSV
    tire_matrix.to_csv('/Users/judahkrug/Desktop/F1-Data/tire_metrics.csv', index=False)

    print("Execution Done")


if __name__ == "__main__":
    main()
