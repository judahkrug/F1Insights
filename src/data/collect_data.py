import pandas as pd

from src.utils.helpers import extract_sc_vsc_periods, get_races, load_race, is_valid_lap, calculate_baseline


def collect_data(years):
    tire_matrix = pd.DataFrame(columns=[
        'Driver', 'Race', 'Year', 'LapNumber', 'Stint', 'StintLapNumber', 'LapTime', 'Compound', 'BaselineTime',
        'DegradationPct', 'SmoothedDeg', 'PositionsGained', 'RacePoints', 'StintLength', 'FinishPosition',
        'StartingPosition'
    ])

    for year in years:
        races = get_races(year)
        yearly_tire_matrix = pd.DataFrame(columns=tire_matrix.columns)

        # Populate Tire Matrix
        yearly_tire_matrix = populate_tire_matrix(year, races, yearly_tire_matrix)

        # Append to the main tire matrix
        tire_matrix = pd.concat([tire_matrix, yearly_tire_matrix], ignore_index=True)

    return tire_matrix


def populate_tire_matrix(year, races, tire_matrix):
    for index, race in enumerate(races):
        session = load_race(year, race[0], 'R')

        print(f"Loading race {index + 1} / {len(races)}")
        print("Loading " + race[0])
        print()

        sc_vsc_periods = extract_sc_vsc_periods(session.track_status)

        driver_points = session.results.set_index('Abbreviation')['Points'].to_dict()

        starting_positions = session.results.set_index('Abbreviation')['GridPosition'].to_dict()
        finish_positions = session.results.set_index('Abbreviation')['Position'].to_dict()

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
                    if is_valid_lap(index, stint_laps, sc_vsc_periods):
                        lap_times.append(stint_laps.LapTime[index])
                        valid_indices.append(index)

                # Skip stints with 1 or fewer valid laps
                if len(valid_indices) <= 1:
                    continue

                # Calculate baseline time
                baseline_time = calculate_baseline(stint_laps, valid_indices)

                # Calculate degradation
                degradation_pcts = [(lap_time - baseline_time) / baseline_time * 100
                                    for lap_time in lap_times]

                # Calculate smoothed degradation using 3-lap rolling average
                smoothed_deg = pd.Series(degradation_pcts).rolling(window=3, min_periods=1).mean()

                stint_length = stint_laps.shape[0]

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
                    race_points = driver_points.get(driver, 0)

                    finish_position = finish_positions.get(driver, 1 + max(finish_positions.values()))
                    starting_position = starting_positions.get(driver)

                    stint_records.append({
                        'Driver': driver,
                        'Race': race[0],
                        'Year': year,
                        'LapNumber': stint_laps.LapNumber[index],
                        'Stint': stint_num,
                        'StintLapNumber': stint_lap_number,
                        'LapTime': lap_time_ms,
                        'Compound': stint_laps.Compound[index],
                        'BaselineTime': baseline_time_ms,
                        'DegradationPct': degradation_pcts[i],
                        'SmoothedDeg': smoothed_deg[i],
                        'PositionsGained': positions_gained,
                        'RacePoints': race_points,
                        'StintLength': stint_length,
                        'FinishPosition': finish_position,
                        'StartingPosition': starting_position
                    })

                # Add stint records to tire_matrix
                stint_df = pd.DataFrame.from_records(stint_records)
                if not tire_matrix.empty:
                    tire_matrix = pd.concat([tire_matrix, stint_df], ignore_index=True)
                else:
                    tire_matrix = stint_df

    return tire_matrix
