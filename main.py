import pandas as pd

from utils import enable_cache, load_race


def main():
    enable_cache()
    session = load_race(2020, 'Austrian Grand Prix', 'R')
    columns = ["Driver", "Position", "LapNumber", "LapTime", "Compound", "TyreLife"]
    laps = session.laps[columns].to_numpy()

    # Print column headers
    print(columns)

    # Print the 2D array
    for lap in laps:
        print(lap)

    print("Done!")

if __name__ == "__main__":
    main()