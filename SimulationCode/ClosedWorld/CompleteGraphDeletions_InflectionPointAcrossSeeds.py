import random
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

""" Perhaps the number of agents needs to vary? Or fix this by doing proportions rather than num of edges """
# GLOBAL PARAMS
NUM_ROUNDS = 100
NUM_AGENTS = 16

def check_if_rows_are_zero(matrix):
    counter = 0
    for row in matrix:
        if np.sum(row) == 0:
            matrix[counter,counter] = 1
        counter += 1
    return matrix

# Gini Coefficient Calculator
def gini_coefficient(values):
    array = np.array(values)
    # Avoid negative or NaN values
    array = array[array >= 0]
    if len(array) == 0:
        return 0.0
    mean = np.mean(array)
    if mean == 0:
        return 0.0
    diff_sum = np.sum(np.abs(array[:, None] - array[None, :]))
    return diff_sum / (2 * len(array)**2 * mean)

def run_simulation(G, dist):
    
    matrix = nx.to_numpy_array(G)
    matrix = check_if_rows_are_zero(matrix)
    # Normalize rows
    matrix = matrix/np.sum(matrix, axis=1)

    ginis: list[float] = []

    for _ in range(NUM_ROUNDS):
        ginis.append(gini_coefficient(dist))
        # Distribution logic
        dist = np.matmul(dist, matrix)
    
    final_gini = np.round(gini_coefficient(dist), decimals=3)
    
    return final_gini, ginis

total_final_ginis = []
total_rounds_to_plot = [i for i in range(0, 120)]
inflection_points_for_diff_seeds = []
# Ensures uniqueness in seed integers
seeds = random.sample(range(1, 401), 50)

# For Complete Graph
for seed in seeds:
    random.seed(seed)
    for j in range(10):
        final_ginis = []
        G = nx.complete_graph(NUM_AGENTS)
        total_edges = len(G.edges())
        pos = nx.circular_layout(G)
        dist = np.zeros(NUM_AGENTS)
        dist[0] = 1
        for i in range(total_edges):
            edge_to_remove = random.choice(list(G.edges()))
            G.remove_edge(*edge_to_remove)
            final_gini, ginis = run_simulation(G, dist)
            final_ginis.append(final_gini)

        total_final_ginis.append(final_ginis)

    column_means_of_ginis = np.mean(total_final_ginis, axis=0)

    x = np.asarray(total_rounds_to_plot)
    y = np.asarray(column_means_of_ginis)
    xx = np.linspace(x.min(), x.max(), 500)

    """ FIT A LINE WITH 4-PARAMETER LOGISTIC """

    def logistic4(x, A, K, k, x0):
        return A + (K - A) / (1 + np.exp(-k * (x - x0)))

    # sensible guesses; clamp A≈0, K≈1 with bounds
    p0 = [0.0, 1.0, 0.3, 95]          # A, K, k, x0
    bounds = ([0.0, 0.7, 0.0, 50],     # lower bounds
            [0.1, 1.05, 5.0, 130])   # upper bounds

    FourParams, _ = curve_fit(logistic4, x, y, p0=p0, bounds=bounds, maxfev=10000)
    A, K, k, x0 = FourParams

    print(f'Inflection Point: {x0}')
    inflection_points_for_diff_seeds.append(x0)

plt.violinplot(inflection_points_for_diff_seeds)
plt.title("Layout of Data")
plt.xticks([])
plt.show()
df = pd.DataFrame(inflection_points_for_diff_seeds)
print(df.describe())

"""
OUTPUT:

count   20.000000
mean   104.387375
std      0.333049
min    104.078194
25%    104.191778
50%    104.296842
75%    104.375167
max    105.459581
"""