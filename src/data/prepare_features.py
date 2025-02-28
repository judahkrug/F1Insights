def prepare_features(df):
    # Aggregate by driver and race
    driver_race_stats = df.groupby(['Driver', 'Race']).agg({
        'SmoothedDeg': ['mean', 'max', 'std'],
        'LapTime': ['mean', 'std'],
        'DegradationPct': ['mean', 'max'],
        'RacePoints': 'max',
        'StintLength': 'mean'
    }).reset_index()

    # Flatten column names
    driver_race_stats.columns = ['Driver', 'Race'] + [
        f'{col[0]}_{col[1]}' for col in driver_race_stats.columns[2:]
    ]

    driver_race_stats.rename(columns={'RacePoints_max': 'RacePoints', 'StintLength_mean': 'StintLength'}, inplace=True)

    return driver_race_stats
