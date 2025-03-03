import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_best_driver(tire_matrix, weighted=True):
    # Group data by Driver
    driver_stats = tire_matrix.groupby('Driver').agg({
        'RacePoints': 'sum',
        'SmoothedDeg': 'mean',  # Lower is better
        'DegradationPct': 'mean',  # Lower is better
        'LapTime': 'mean',  # Lower is better
        'PositionsGained': 'sum',  # Higher is better
        'Stint': 'count',
        'FinishPosition': 'mean'  # Lower is better
    }).reset_index()

    # Calculate races participated
    races_by_driver = tire_matrix.groupby('Driver')['Race'].nunique().reset_index()
    races_by_driver.rename(columns={'Race': 'RacesParticipated'}, inplace=True)
    driver_stats = pd.merge(driver_stats, races_by_driver, on='Driver')

    # Remove drivers where RacesParticipated < 10
    driver_stats = driver_stats[driver_stats['RacesParticipated'] >= 10].reset_index()

    # Calculate points per race
    driver_stats['PointsPerRace'] = driver_stats['RacePoints'] / driver_stats['RacesParticipated']

    # Calculate positions gained per race
    driver_stats['PositionsGainedPerRace'] = driver_stats['PositionsGained'] / driver_stats['RacesParticipated']

    # Calculate tire management score (inverse of degradation - lower degradation is better)
    driver_stats['TireManagementScore'] = -1 * driver_stats['SmoothedDeg']

    scaler = MinMaxScaler()

    # For lap time, we need to invert the scaling (faster lap times are better)
    driver_stats['LapTime_Normalized'] = -1 * scaler.fit_transform(
        driver_stats[['LapTime']]
    )
    driver_stats['FinishPosition_Normalized'] = -1 * scaler.fit_transform(
        driver_stats[['FinishPosition']]
    )

    # Normalize other metrics
    normalized_features = pd.DataFrame(
        scaler.fit_transform(driver_stats[['PointsPerRace', 'PositionsGainedPerRace', 'TireManagementScore']]),
        columns=[f"{col}_Normalized" for col in ['PointsPerRace', 'PositionsGainedPerRace', 'TireManagementScore']]
    )

    # Add normalized features to driver stats
    for col in ['PointsPerRace', 'PositionsGainedPerRace', 'TireManagementScore']:
        driver_stats[f"{col}_Normalized"] = normalized_features[f"{col}_Normalized"]


    # Apply weights to different metrics if weighted is True
    if weighted:
        weights = {
            'PointsPerRace_Normalized': 0.4,  # Race points are important
            'TireManagementScore_Normalized': 0.3,  # Tire management is critical
            'FinishPosition_Normalized': 0.2,  # Finishing position is important
            'PositionsGainedPerRace_Normalized': 0.1  # Positions gained shows overtaking ability
        }
    else:
        # Equal weights
        weights = {col: 1 / 4 for col in [
            'PointsPerRace_Normalized',
            'TireManagementScore_Normalized',
            'FinishPosition_Normalized',
            'PositionsGainedPerRace_Normalized'
        ]}

    # Calculate composite score
    driver_stats['CompositeScore'] = sum(
        driver_stats[metric] * weight
        for metric, weight in weights.items()
    )

    # Rank drivers by composite score
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
        sns.barplot(x='PointsPerRace', y='Driver', data=top_drivers)
        plt.title(f'Top {top_n} Drivers by Points Per Race')
        plt.xlabel('Points Per Race')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Average Finish Position
        plt.figure(figsize=(12, 8))
        sns.barplot(x='FinishPosition', y='Driver', data=top_drivers)
        plt.title(f'Top {top_n} Drivers by Average Finish Position (Lower is Better)')
        plt.xlabel('Average Finish Position')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

        # Tire management
        plt.figure(figsize=(12, 8))
        sns.barplot(x='SmoothedDeg', y='Driver', data=top_drivers)
        plt.title(f'Top {top_n} Drivers by Tire Degradation (Lower is Better)')
        plt.xlabel('Average Tire Degradation %')
        plt.ylabel('Driver')
        plt.tight_layout()
        plt.show()

    else:
        # Create radar charts for top drivers
        metrics = [
            'PointsPerRace_Normalized',
            'TireManagementScore_Normalized',
            'FinishPosition_Normalized',
            'PositionsGainedPerRace_Normalized'
        ]

        metric_labels = [
            'Points Per Race',
            'Tire Management',
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