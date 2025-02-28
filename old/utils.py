from fastf1.ergast import Ergast


def add_points_to_matrix(tire_matrix, year):
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=year, round="last", result_type="raw")
    standings = standings[0]['DriverStandings']

    driver_points = {}

    for entry in standings:
        driver_code = entry['Driver']['code']
        points = entry.get('points', 0)  # Set points to 0 if not available
        driver_points[driver_code] = points

    tire_matrix[str(year) + ' Points'] = tire_matrix['Driver'].map(driver_points).fillna(0)
    tire_matrix.to_csv('/Users/judahkrug/Desktop/F1-Data/' + str(year) + '_multipliers_with_points.csv', index=False)
