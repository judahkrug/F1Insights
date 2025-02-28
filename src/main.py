import pandas as pd

from src.data.collect_data import collect_data
from src.models.train_model import train_and_evaluate_model
from src.utils.helpers import enable_cache
from src.data.prepare_features import prepare_features


def main():
    enable_cache()
    years = [2022, 2023, 2024]

    # Data collection - with option to skip if already collected
    collect_new_data = True  # Set to True to collect fresh data
    if collect_new_data:
        tire_matrix = collect_data(years)
        tire_matrix.to_csv('data/processed/tire_metrics.csv', index=False)
    else:
        tire_matrix = pd.read_csv('data/processed/tire_metrics.csv')


    # Prepare features
    modeling_data = prepare_features(tire_matrix)

    # Split into train and test
    train_data = modeling_data[modeling_data['Race'].str.contains('2022|2023')].copy()
    test_data = modeling_data[modeling_data['Race'].str.contains('2024')].copy()

    # Define enhanced predictors
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

    # Define target (using RacePoints as it's a better measure of performance)
    # target = 'RacePoints'
    target = 'StintLength'

    # Train and evaluate model
    reg, scaler, feature_importance = train_and_evaluate_model(train_data, test_data, predictors, target)

    return reg, scaler, feature_importance


if __name__ == "__main__":
    reg, scaler, importance = main()
    print("\nExecution Done")
