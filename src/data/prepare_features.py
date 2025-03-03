import pandas as pd


def prepare_features(df):
    # Group by driver and race
    driver_race_stats = df.groupby(['Driver', 'Race', 'Compound']).agg({
        'SmoothedDeg': ['mean', 'max', 'std'],
        'LapTime': ['mean', 'std', 'min'],
        'DegradationPct': ['mean', 'max', 'median'],
        'RacePoints': 'max',
        'StintLength': 'mean',
        'PositionsGained': 'sum'
    }).reset_index()

    # Flatten columns
    driver_race_stats.columns = ['Driver', 'Race', 'Compound'] + [
        f'{col[0]}_{col[1]}' for col in driver_race_stats.columns[3:]
    ]

    # Create tire-specific features
    tire_dummies = pd.get_dummies(driver_race_stats['Compound'], prefix='Tire')
    driver_race_stats = pd.concat([driver_race_stats, tire_dummies], axis=1)

    # Add a relative performance metric
    driver_race_stats['RelativePerformance'] = driver_race_stats['LapTime_min'] / driver_race_stats.groupby('Race')[
        'LapTime_min'].transform('mean')

    driver_race_stats.rename(columns={
        'RacePoints_max': 'RacePoints',
        'StintLength_mean': 'StintLength',
        'PositionsGained_sum': 'PositionsGained'
    }, inplace=True)

    return driver_race_stats
