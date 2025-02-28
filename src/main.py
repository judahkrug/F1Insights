import pandas as pd

from src.models.train_model import train_and_evaluate_model
from src.utils.helpers import enable_cache
from src.data.prepare_features import prepare_features


def main():
    enable_cache()
    years = [2022, 2023, 2024]

    # ---- Uncomment the below lines to collect data ----
    # tire_matrix = collect_data(years)
    # tire_matrix.to_csv('/Users/judahkrug/Desktop/F1-Data/tire_metrics.csv', index=False)

    tire_matrix = pd.read_csv('/Users/judahkrug/Desktop/F1-Data/tire_metrics.csv')

    # Prepare features
    modeling_data = prepare_features(tire_matrix)

    # Split into train and test
    train_data = modeling_data[modeling_data['Race'].str.contains('2022|2023')].copy()
    test_data = modeling_data[modeling_data['Race'].str.contains('2024')].copy()

    # Define predictors (removing highly correlated features)
    predictors = [
        'SmoothedDeg_mean',
        'SmoothedDeg_std',
        'LapTime_mean',
        'LapTime_std',
        'DegradationPct_max'
    ]

    # Define target (using RacePoints as it's a better measure of performance)
    # target = 'RacePoints'
    target = 'StintLength'

    # Train and evaluate model
    reg, scaler, feature_importance = train_and_evaluate_model(train_data, test_data, predictors, target)

    return reg, scaler, feature_importance


if __name__ == "__main__":
    reg, scaler, importance = main()
    print("\nExecution Done")
