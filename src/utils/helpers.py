import fastf1
import numpy as np
import pandas as pd


def enable_cache(cache_dir='cache'):
    fastf1.Cache.enable_cache(cache_dir)


def load_race(year, grand_prix, session):
    race = fastf1.get_session(year, grand_prix, session)
    race.load()
    return race


def get_races(year):
    events = fastf1.get_event_schedule(year)
    races = []
    for i in range(events.shape[0]):
        if events['Session5'][i] == 'Race':
            race_info = (events['OfficialEventName'][i], events['Country'][i])
            races.append(race_info)
    return races


def is_pit_lap(index, laps):
    if pd.notnull(laps.PitInTime[index]) or pd.notnull(laps.PitOutTime[index]):
        return True
    return False


def is_valid_lap(index, stint_laps, sc_vsc_periods=None):
    if (stint_laps.LapNumber[index] == 1 or
            is_pit_lap(index, stint_laps) or
            pd.isnull(stint_laps.LapTime[index]) or
            stint_laps.Deleted[index]):
        return False

    # If SC/VSC periods are provided, check if lap was affected
    if sc_vsc_periods:
        lap_start_time = stint_laps.loc[index, 'LapStartTime']
        lap_end_time = stint_laps.loc[index, 'Time']

        # Check if lap overlaps with any SC/VSC period
        for start, end in sc_vsc_periods:
            if not ((lap_end_time < start) or (lap_start_time > end)):
                return False

    return True


def calculate_baseline(stint_laps, valid_indices):
    # Get the first 3 valid laps of the stint
    initial_valid_laps = stint_laps.loc[valid_indices[:3]].copy()

    if len(initial_valid_laps) == 0:
        return np.nan

    # Calculate median and standard deviation of these valid laps
    median_time = initial_valid_laps['LapTime'].median()
    std_time = initial_valid_laps['LapTime'].std()

    # If first valid lap is an outlier (> 1.5 std from median), use second best lap as baseline
    if abs(initial_valid_laps.iloc[0]['LapTime'] - median_time) > 1.5 * std_time:
        # Use the fastest non-outlier lap from first 3 valid laps as baseline
        valid_laps = initial_valid_laps[
            abs(initial_valid_laps['LapTime'] - median_time) <= 1.5 * std_time
            ]
        if len(valid_laps) > 0:
            baseline = valid_laps['LapTime'].min()
        else:
            # If all initial valid laps are outliers, use the median
            baseline = median_time
    else:
        baseline = initial_valid_laps.iloc[0]['LapTime']

    return baseline


def extract_sc_vsc_periods(track_status):
    sc_vsc_periods = []
    sc_vsc_active = False
    start_time = None

    for i, row in track_status.iterrows():
        if 'SCDeployed' in row['Message'] or 'VSCDeployed' in row['Message']:
            sc_vsc_active = True
            start_time = row['Time']
        elif 'AllClear' in row['Message'] and sc_vsc_active:
            sc_vsc_active = False
            sc_vsc_periods.append((start_time, row['Time']))

    # If SC/VSC is still active at the end of the data
    if sc_vsc_active and start_time is not None:
        sc_vsc_periods.append((start_time, pd.Timedelta.max))

    return sc_vsc_periods
