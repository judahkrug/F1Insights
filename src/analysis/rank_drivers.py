import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    points_per_race_weight,
    tire_management_score_weight,
    starting_position_weight,
    finish_position_weight,
    avg_positions_gained_weight
)

def analyze_best_driver(tire_matrix, weighted=True):
    # First calculate RacePoints correctly - sum points per unique Race-Driver combination
    race_points = tire_matrix.drop_duplicates(subset=['Driver', 'Race'])[['Driver', 'Race', 'RacePoints']]
    total_points_by_driver = race_points.groupby('Driver')['RacePoints'].sum().reset_index()

    finish_position = tire_matrix.drop_duplicates(subset=['Driver', 'Race'])[['Driver', 'Race', 'FinishPosition']]
    avg_finish_position_by_driver = finish_position.groupby('Driver')['FinishPosition'].mean().reset_index()

    starting_position = tire_matrix.drop_duplicates(subset=['Driver', 'Race'])[['Driver', 'Race', 'StartingPosition']]
    avg_starting_position_by_driver = starting_position.groupby('Driver')['StartingPosition'].mean().reset_index()

    # Group data by Driver for other metrics
    driver_stats = tire_matrix.groupby('Driver').agg({
        'SmoothedDeg': 'mean',  # Lower is better
        'DegradationPct': 'mean',  # Lower is better
        'Stint': 'count'
    }).reset_index()

    # Calculate races participated
    races_by_driver = tire_matrix.groupby('Driver')['Race'].nunique().reset_index()
    races_by_driver.rename(columns={'Race': 'RacesParticipated'}, inplace=True)

    # Merge data frames
    driver_stats = pd.merge(driver_stats, races_by_driver, on='Driver')
    driver_stats = pd.merge(driver_stats, total_points_by_driver, on='Driver')
    driver_stats = pd.merge(driver_stats, avg_finish_position_by_driver, on='Driver')
    driver_stats = pd.merge(driver_stats, avg_starting_position_by_driver, on='Driver')

    # Remove drivers where RacesParticipated < 10
    driver_stats = driver_stats[driver_stats['RacesParticipated'] >= 10].reset_index()

    # Calculate new columns
    driver_stats['PointsPerRace'] = driver_stats['RacePoints'] / driver_stats['RacesParticipated']
    driver_stats['AvgPositionsGained'] = driver_stats['StartingPosition'] - driver_stats['FinishPosition']
    driver_stats['TireManagementScore'] = -1 * driver_stats['SmoothedDeg']

    scaler = MinMaxScaler()

    driver_stats['FinishPosition_Normalized'] = 1 + (-1 * scaler.fit_transform(
        driver_stats[['FinishPosition']]
    ))

    driver_stats['StartingPosition_Normalized'] = 1 + (-1 * scaler.fit_transform(
        driver_stats[['StartingPosition']]
    ))

    # Normalize other metrics
    normalized_features = pd.DataFrame(
        scaler.fit_transform(driver_stats[['PointsPerRace', 'AvgPositionsGained', 'TireManagementScore']]),
        columns=[f"{col}_Normalized" for col in ['PointsPerRace', 'AvgPositionsGained', 'TireManagementScore']]
    )

    # Add normalized features to driver stats
    for col in ['PointsPerRace', 'AvgPositionsGained', 'TireManagementScore']:
        driver_stats[f"{col}_Normalized"] = normalized_features[f"{col}_Normalized"]

    # Apply weights to different metrics if weighted is True
    weights = {
        'PointsPerRace_Normalized': points_per_race_weight,
        'TireManagementScore_Normalized': tire_management_score_weight,
        'StartingPosition_Normalized': starting_position_weight,
        'FinishPosition_Normalized': finish_position_weight,
        'AvgPositionsGained_Normalized': avg_positions_gained_weight
    } if weighted else {col: 0.2 for col in [
        'PointsPerRace_Normalized',
        'TireManagementScore_Normalized',
        'StartingPosition_Normalized',
        'FinishPosition_Normalized',
        'AvgPositionsGained_Normalized'
    ]}

    # Calculate composite score
    driver_stats['CompositeScore'] = sum(
        driver_stats[metric] * weight for metric, weight in weights.items()
    )
    # Rank drivers
    ranked_drivers = driver_stats.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
    ranked_drivers.index = ranked_drivers.index + 1  # Start index at 1

    return ranked_drivers


def plot_driver_rankings(ranked_drivers, top_n=10, radar=False):
    # Get top N drivers
    top_drivers = ranked_drivers.head(top_n)

    if not radar:
        # Bar chart of composite scores
        plt.figure(figsize=(12, 8))
        sns.barplot(x='CompositeScore', y='Driver', data=top_drivers)
        plt.title(f'Top {top_n} Drivers by Composite Performance Score')
        plt.xlabel('Composite Performance Score')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Points per race
        plt.figure(figsize=(12, 8))
        points_sorted = ranked_drivers.sort_values('PointsPerRace', ascending=False).head(top_n)
        sns.barplot(x='PointsPerRace', y='Driver', data=points_sorted)
        plt.title(f'Top {top_n} Drivers by Points Per Race')
        plt.xlabel('Points Per Race')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Positions gained per race
        plt.figure(figsize=(12, 8))
        positions_sorted = ranked_drivers.sort_values('AvgPositionsGained', ascending=False).head(top_n)
        sns.barplot(x='AvgPositionsGained', y='Driver', data=positions_sorted)
        plt.title(f'Top {top_n} Drivers by Positions Gained Per Race')
        plt.xlabel('Positions Gained Per Race')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Average Starting Position
        plt.figure(figsize=(12, 8))
        starting_sorted = ranked_drivers.sort_values('StartingPosition', ascending=True).head(top_n)
        sns.barplot(x='StartingPosition', y='Driver', data=starting_sorted)
        plt.title(f'Top {top_n} Drivers by Average Starting Position')
        plt.xlabel('Average Starting Position')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Average Finish Position
        plt.figure(figsize=(12, 8))
        finish_sorted = ranked_drivers.sort_values('FinishPosition', ascending=True).head(top_n)
        sns.barplot(x='FinishPosition', y='Driver', data=finish_sorted)
        plt.title(f'Top {top_n} Drivers by Average Finish Position')
        plt.xlabel('Average Finish Position')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Tire management
        plt.figure(figsize=(12, 8))
        tire_sorted = ranked_drivers.sort_values('SmoothedDeg', ascending=False).head(top_n)
        sns.barplot(x='SmoothedDeg', y='Driver', data=tire_sorted)
        plt.title(f'Top {top_n} Drivers by Tire Degradation')
        plt.xlabel('Average Tire Degradation %')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

    else:
        # Create radar charts for top drivers
        metrics = [
            'PointsPerRace_Normalized',
            'TireManagementScore_Normalized',
            'StartingPosition_Normalized',
            'FinishPosition_Normalized',
            'AvgPositionsGained_Normalized'
        ]

        metric_labels = [
            'Points Per Race',
            'Tire Management',
            'Starting Position',
            'Finish Position',
            'Positions Gained'
        ]

        # Number of variables
        N = len(metrics)

        # Create subplot figure with 3 rows and 3 columns
        fig, axes = plt.subplots(nrows=min(3, (top_n + 2) // 3), ncols=3, figsize=(15, 15),
                                 subplot_kw=dict(polar=True))
        axes = axes.flatten()

        for i, driver in enumerate(top_drivers['Driver'].values[:top_n]):
            if i >= len(axes):
                break

            # Get driver data
            driver_data = top_drivers.loc[top_drivers['Driver'] == driver, metrics].values.flatten().tolist()

            # Close the plot by appending the first value
            values = driver_data + [driver_data[0]]

            # Angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Plot data
            ax = axes[i]
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.1)

            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)

            # Draw ylabels
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 1)

            # Add title
            ax.set_title(f"{driver} (Rank {i + 1})", size=11, y=1.1)

        # Hide unused subplots
        for i in range(min(top_n, len(axes)), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
