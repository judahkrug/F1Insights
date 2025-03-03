import pandas as pd

from src.analysis.rank_drivers import analyze_best_driver, plot_driver_rankings
from src.data.collect_data import collect_data
from src.analysis.train_model import train_and_evaluate_model
from src.utils.helpers import enable_cache
from src.data.prepare_features import prepare_features
from src.config import collect_new_data, years, train_years, test_years, use_weights
from pathlib import Path


def main():
    enable_cache()
    data_path = Path('data/processed/tire_metrics.csv')

    if collect_new_data:
        tire_matrix = collect_data(years)
        tire_matrix.to_csv(data_path, index=False)
    else:
        tire_matrix = pd.read_csv(data_path)

    # Prepare features
    modeling_data = prepare_features(tire_matrix)

    # Split into train and test
    train_data = modeling_data[modeling_data['Race'].str.contains('|'.join(map(str, train_years)))].copy()
    test_data = modeling_data[modeling_data['Race'].str.contains('|'.join(map(str, test_years)))].copy()

    # Define predictors
    predictors = [
        'SmoothedDeg_mean',
        'SmoothedDeg_std',
        'LapTime_mean',
        'LapTime_std',
        'LapTime_min',
        'DegradationPct_mean',
        'DegradationPct_max',
        'DegradationPct_median',
        'RelativePerformance',
        'PositionsGained'
    ]

    # Add tire compound dummy variables
    tire_columns = [col for col in modeling_data.columns if col.startswith('Tire_')]
    predictors.extend(tire_columns)

    # Define target
    target_stint_length = 'StintLength'
    target_race_points = 'RacePoints'

    # Train and evaluate model
    train_and_evaluate_model(train_data, test_data, predictors, target_stint_length)
    train_and_evaluate_model(train_data, test_data, predictors, target_race_points)

    # NEW: Analyze best drivers
    print("====================")
    print("BEST DRIVER ANALYSIS")
    print("====================")

    # Overall driver rankings
    driver_rankings = analyze_best_driver(tire_matrix, weighted=use_weights)

    # Display top 20 drivers
    print("\nTop 20 Drivers (Overall Performance):")
    print(driver_rankings[['Driver', 'CompositeScore', 'PointsPerRace',
                           'DegradationPct', 'AvgPositionsGained']].head(20))

    # Visualize the driver rankings
    plot_driver_rankings(driver_rankings, top_n=20)

    # Visualize top drivers with radar charts
    plot_driver_rankings(driver_rankings, top_n=6, radar=True)


if __name__ == "__main__":
    main()
    print()
    print("Execution Done")
