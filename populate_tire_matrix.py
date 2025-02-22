import pandas as pd
import numpy as np

from utils import calculate_baseline, is_pit_lap, is_valid_lap, load_race


def populate_tire_matrix(year, races, tire_matrix):
    for race in races:

        session = load_race(year, race[0], 'R')

        print(f"Loading race {race[2]} / {len(races)}")
        print("Loading " + race[0])
        print()

        for driver in session.results.Abbreviation.values:
            laps = session.laps.pick_drivers(driver)

            if laps.Position.empty:
                continue
        
            # Process each stint separately
            for stint_num in laps.Stint.unique():
                stint_laps = laps[laps.Stint == stint_num]
                lap_times = []
                valid_indices = []
                stint_records = []

                # Collect valid lap data
                for index, lapNumber in stint_laps.LapNumber.items():
                    if is_valid_lap(index, stint_laps):
                        lap_times.append(stint_laps.LapTime[index])
                        valid_indices.append(index)

                # Skip empty stints
                if not lap_times:
                    continue

                # Calculate baseline time
                baseline_time = calculate_baseline(stint_laps, valid_indices)

                # Calculate degradation
                degradation_pcts = [(lap_time - baseline_time) / baseline_time * 100
                                    for lap_time in lap_times]

                # Calculate smoothed degradation using 3-lap rolling average
                smoothed_deg = pd.Series(degradation_pcts).rolling(window=3, min_periods=1).mean()

                # Populate tire_matrix with processed data
                for i, index in enumerate(valid_indices):
                    positions_gained = 0
                    if i > 0:  # If not the first valid lap
                        prev_index = valid_indices[i - 1]
                        positions_gained = stint_laps.Position[index] - stint_laps.Position[prev_index]

                    # Get the lap number within this stint
                    stint_lap_number = (stint_laps.loc[index, 'LapNumber'] -
                                        stint_laps['LapNumber'].iloc[0] + 1)

                    # Convert LapTime and BaselineTime to milliseconds
                    lap_time_ms = stint_laps.LapTime[index].total_seconds() * 1000
                    baseline_time_ms = baseline_time.total_seconds() * 1000

                    stint_records.append({
                        'Driver': driver,
                        'Race': race[0],
                        'LapNumber': stint_laps.LapNumber[index],
                        'Stint': stint_num,
                        'StintLapNumber': stint_lap_number,
                        'LapTime': lap_time_ms,
                        'Compound': stint_laps.Compound[index],
                        'BaselineTime': baseline_time_ms,
                        'DegradationPct': degradation_pcts[i],
                        'SmoothedDeg': smoothed_deg[i],
                        'PositionsGained': positions_gained
                    })

                # Add stint records to tire_matrix
                stint_df = pd.DataFrame.from_records(stint_records)
                tire_matrix = pd.concat([tire_matrix, stint_df], ignore_index=True)

    return tire_matrix