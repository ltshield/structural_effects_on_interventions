import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.integrate import quad
import pandas as pd

# GLOBAL PARAMS

# SEED = 42
NUM_ROUNDS = 100
NUM_AGENTS = 16
VMIN = 0
VMAX = 1.0

# random.seed(SEED)

def transition_metrics(L, U, k, x0, m):
    x10 = x0 + (1/k)*np.log(0.1/0.9)
    x90 = x0 + (1/k)*np.log(0.9/0.1)
    width = x90 - x10
    slope = k*(U-L)/4
    area, _ = quad(lambda x: L + (U-L)/(1+np.exp(-k*(x-x0))), 0, m)
    A = area/m
    R = 1 - A
    p0 = 1 - x0/m
    return dict(x0=x0, width=width, slope=slope, A=A, R=R, p0=p0)

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

def run_simulation(G, dist, print_final_gini=False, graph_gini_progress=False, show_iterations=False):
    
    matrix = nx.to_numpy_array(G)
    matrix = check_if_rows_are_zero(matrix)
    # Normalize rows
    matrix = matrix/np.sum(matrix, axis=1)

    ginis: list[float] = []
    rounds = [i for i in range(NUM_ROUNDS)]

    if show_iterations:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10,8))
        node_order = [i for i in range(NUM_AGENTS)]
        
        for i in range(NUM_ROUNDS):

            ginis.append(gini_coefficient(dist))

            prob_dict = {node_order[i]: dist[i] for i in range(len(node_order))}
            nx.set_node_attributes(G, prob_dict, name="token_prob")
            ax.cla()

            pos = nx.circular_layout(G)

            probs = [G.nodes[n]["token_prob"] for n in G.nodes]

            nx.draw_networkx_nodes(G, pos,
                                        node_color=probs,
                                        cmap=cm['viridis'],
                                        vmin=VMIN,
                                        vmax=VMAX,
                                        node_size=1000
            )

            nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=30, edge_color='gray')

            labels = {n: f"{n}\n{G.nodes[n]['token_prob']:.2f}" for n in G.nodes}

            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="white")

            plt.title(f"Round {i+1}")
            plt.pause(1)

            dist = np.matmul(dist, matrix)

    else:
        for _ in range(NUM_ROUNDS):
            ginis.append(gini_coefficient(dist))
            # Distribution logic
            dist = np.matmul(dist, matrix)
    
    if graph_gini_progress:
        plt.plot(rounds, ginis, color="blue", label="Scale-free Network")
        plt.xlabel("Round")
        plt.ylabel("Gini Coefficient")
        plt.show()

    if print_final_gini:
        print(f"Final Gini Coefficient: {np.round(gini_coefficient(dist), decimals=3)}")
    
    final_gini = np.round(gini_coefficient(dist), decimals=3)
    
    return final_gini, ginis

def check_if_rows_are_zero(matrix):
    counter = 0
    for row in matrix:
        if np.sum(row) == 0:
            matrix[counter,counter] = 1
        counter += 1
    return matrix

total_params = []
seeds = random.sample(range(1, 1001), 100)

for f in range(100):
    random.seed(seeds[f])
    total_final_ginis = []
    total_rounds_to_plot = [i for i in range(0, 120)]

    final_ginis = []
    G = nx.complete_graph(NUM_AGENTS)
    total_edges = len(G.edges())
    pos = nx.circular_layout(G)
    dist = np.zeros(NUM_AGENTS)
    dist[0] = 1
    order_of_deletions = []
    for i in range(total_edges):
        edge_to_remove = random.choice(list(G.edges()))
        order_of_deletions.append(edge_to_remove)
        # apparently * will unpack into u, v for me
        G.remove_edge(*edge_to_remove)
        final_gini, ginis = run_simulation(G, dist)
        final_ginis.append(final_gini)

    total_final_ginis.append(final_ginis)

    plt.scatter([i for i in range(0,total_edges)], final_ginis, alpha=0.3, s=30)

    column_means_of_ginis = np.mean(total_final_ginis, axis=0)

    # Fit a line to scatter plot using polyfit of degree 4 from 75 deletions and up
    # coeffs = np.polyfit([i for i in range(0,total_edges)], column_means_of_ginis, 4)
    # poly_eq = np.poly1d(coeffs)
    # plt.plot([i for i in range(0,total_edges)][75:], poly_eq([i for i in range(0,total_edges)][75:]))

    """ 

    Given that the gini coefficient is bounded at 0 and 1, linear and exponential models are not of use to us. Therefore,
    we will use sigmoid-type models which show us system transitions and handle bounded growth like that of the Gini Coeff.
        
    """

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

    total_params.append(FourParams)

    # print(f'Inflection Point: {x0}')

results = [transition_metrics(L, U, k, x0, total_edges)
        for L, U, k, x0 in total_params]
df = pd.DataFrame(results)

# summary statistics
summary = df.describe().loc[['mean', 'std', 'min', 'max']]
print(summary)

yyFourParams = logistic4(xx, *np.mean(total_params, axis=0))
plt.plot(xx, yyFourParams, label='Sigmoid fit', linewidth=2)

""" 

FITTING A LINE WITH GOMPERTZ ALGORITHM

def gompertz(x, A, K, b, x0):
    return A + (K - A) * np.exp(-np.exp(-b * (x - x0)))

p0 = [0.0, 1.0, 0.2, 95]
bounds = ([0.0, 0.7, 0.0, 50],
        [0.1, 1.05, 5.0, 130])

# GompertzParams, _ = curve_fit(gompertz, x, y, p0=p0, bounds=bounds, maxfev=10000)
# yyGompertz = gompertz(xx, *GompertzParams)
# plt.plot(xx, yyGompertz, label='Gompertz Fit', linewidth=2)

"""

plt.legend()
plt.xlabel("Number of Edges Deleted")
plt.ylabel("Gini Coefficient After 100 Rounds")
plt.xticks([0,20,40,60,80,100,120])

plt.show()

# Perhaps compare the parameters from both models and compare them in a table in the paper?

""" Error Checking """
# yHatFourParams = logistic4(x, *FourParams)
# r2FourParams = 1 - np.sum((y - yHatFourParams)**2) / np.sum((y - y.mean())**2)
# print(f'R^2 FourParams ≈ {r2FourParams:.3f}')

# yHatGompertz = gompertz(x, *GompertzParams)
# r2Gompertz = 1 - np.sum((y - yHatGompertz)**2) / np.sum((y - y.mean())**2)
# print(f'R^2 Gompertz ≈ {r2Gompertz:.3f}')

# def rmse(y, y_pred):
#     return np.sqrt(np.mean((y - y_pred)**2))

# def aic(n, rss, k):
#     return n * np.log(rss / n) + 2 * k  # k = number of parameters

# rss_log = np.sum((y - yHatFourParams)**2)
# rss_gom = np.sum((y - yHatGompertz)**2)

# print("Logistic RMSE:", rmse(y, yHatFourParams))
# print("Gompertz RMSE:", rmse(y, yHatGompertz))
# print("Logistic AIC:", aic(len(y), rss_log, 4))
# print("Gompertz AIC:", aic(len(y), rss_gom, 4))

""" 

Error Findings:

R^2 FourParams ≈ 0.996
R^2 Gompertz ≈ 0.992
Logistic RMSE: 0.01758940903144635
Gompertz RMSE: 0.0232429606751656
Logistic AIC: -961.7099962347772
Gompertz AIC: -894.8207102683347

Conclusion: 
Logistic is a better fit for seed=42's data.

"""
