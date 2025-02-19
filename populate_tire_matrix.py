import pandas as pd
import numpy as np

from utils import is_pit_lap, load_race


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
        
            initial_position = laps.Position.iloc[0]

            # Process each stint separately
            for stint_num in laps.Stint.unique():
                stint_laps = laps[laps.Stint == stint_num]
                lap_times = []
                valid_indices = []
                stint_records = []

                # TODO: Improve valid lap function
                # Collect valid lap data
                for index, lapNumber in stint_laps.LapNumber.items():
                    if lapNumber != 1 and not is_pit_lap(index, stint_laps) and pd.notnull(stint_laps.LapTime[index]):
                        lap_times.append(stint_laps.LapTime[index])
                        valid_indices.append(index)

                # Skip empty stints
                if not lap_times:
                    continue

                # Calculate degradation
                baseline_time = lap_times[0]
                degradation_pcts = [(lap_time - baseline_time) / baseline_time * 100
                                    for lap_time in lap_times]

                # Calculate smoothed degradation using 3-lap rolling average
                smoothed_deg = pd.Series(degradation_pcts).rolling(window=3, min_periods=1).mean()

                # Populate tire_matrix with processed data
                for i, index in enumerate(valid_indices):
                    positions_gained = initial_position - stint_laps.Position[index]

                    # TODO: Convert LapTime and BaselineTime to milliseconds? Instead of DateTime...

                    stint_records.append({
                        'Driver': driver,
                        'Race': race[0],
                        'Stint': stint_num,
                        'StintLapNumber': stint_laps.LapNumber[index],
                        'LapTime': stint_laps.LapTime[index],
                        'Compound': stint_laps.Compound[index],
                        'BaselineTime': baseline_time,
                        'DegradationPct': degradation_pcts[i],
                        'SmoothedDeg': smoothed_deg[i],
                        'PositionsGained': positions_gained
                    })

                # Add stint records to tire_matrix
                stint_df = pd.DataFrame.from_records(stint_records)
                tire_matrix = pd.concat([tire_matrix, stint_df], ignore_index=True)

    return tire_matrix