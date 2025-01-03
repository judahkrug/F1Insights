import pandas as pd
import fastf1



# Enable Cache
fastf1.Cache.enable_cache('cache')

# Load Race
race = fastf1.get_session(2020, 'Austrian Grand Prix', 'R')
race.load()
print(race.laps)