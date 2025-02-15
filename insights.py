import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_insights(tire_matrix):
    sns.lmplot(x="Average Tire Score", y="Points", data=tire_matrix, fit_reg=True, ci=None)
    plt.show()  # Display the plot
    print("generate_insights")


if __name__ == "__main__":
    tire_matrix = pd.read_csv('/Users/judahkrug/Desktop/multipliers_with_points.csv', index_col=0)
    generate_insights(tire_matrix)