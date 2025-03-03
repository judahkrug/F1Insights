# Configuration settings

# Set to True to collect new data, False to use existing data
collect_new_data = False

# List of years to collect data for
years = [2020, 2021, 2022, 2023, 2024]

# Split train/test data for ML algorithms
train_years = [2020, 2021, 2022, 2023]
test_years = [2024]

# Set to True to use weighted analysis
use_weights = True

# Weights for different metrics in rank_drivers
points_per_race_weight = 0.3
tire_management_score_weight = 0.3
starting_position_weight = 0.2
finish_position_weight = 0.2
avg_positions_gained_weight = 0.0