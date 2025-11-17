import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

"""
Future work, normalize curve and scatter plot to 0-1 (% of deletions so we can compare network types and behaviors)
maybe go off of first point off of the 0 gini coefficient plane? or past a certain threshold value? (average?)
inflection point might not be the most representative point for this problem.
"""
NUM_ROUNDS = 150

ITERATIONS = 10

# # Complete
# COMP_NUM_AGENTS = 16

# # Random
# RAND_NUM_AGENTS = 16
# RAND_PROB = 6/19

# # Small-world
# SW_NUM_AGENTS = 16
# SW_K = 6
# SW_PROB = 0.2

# # Scale-free
# SF_NUM_AGENTS = 16
# SF_M = 3

# Complete
COMP_NUM_AGENTS = 25

# Random
RAND_NUM_AGENTS = 200
RAND_PROB = 0.05

# Small-world
SW_NUM_AGENTS = 200
SW_K = 8
SW_PROB = 0.05

# Scale-free
SF_NUM_AGENTS = 200
SF_M = 3

def check_if_rows_are_zero(matrix):
    counter = 0
    for row in matrix:
        if np.sum(row) == 0:
            matrix[counter,counter] = 1
        counter += 1
    return matrix

# TODO: Does this stay consistent if I did degree-based distribution of resources or other randomly assigned starter node for the token?
# also might need to up the number of rounds if I am using such a large number of nodes to let the tokens properly disperse before checking the final gini

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

def logistic4(x, A, K, k, x0):
        return A + (K - A) / (1 + np.exp(-k * (x - x0)))

def gatherParams(graph_type):
    seeds = random.sample(range(1, 1001), ITERATIONS)

    total_params = []
    for seed in seeds:
        random.seed(seed)

        if graph_type == "COMP":
            G = nx.complete_graph(COMP_NUM_AGENTS)
            total_edges = len(G.edges())
            print(f"COMP Total Edges: {total_edges}")
            dist = np.zeros(COMP_NUM_AGENTS)
            color = "green"
        elif graph_type == "SW":
            G = nx.newman_watts_strogatz_graph(SW_NUM_AGENTS, SW_K, SW_PROB)
            total_edges = len(G.edges())
            print(f"SW Total Edges: {total_edges}")
            dist = np.zeros(SW_NUM_AGENTS)
            color = "blue"
        elif graph_type == "RANDOM":
            G = nx.erdos_renyi_graph(RAND_NUM_AGENTS, RAND_PROB)
            total_edges = len(G.edges())
            print(f"RAND Total Edges: {total_edges}")
            dist = np.zeros(RAND_NUM_AGENTS)
            color = "purple"
        elif graph_type == "SF":
            G = nx.barabasi_albert_graph(SF_NUM_AGENTS, SF_M)
            total_edges = len(G.edges())
            print(f"SF Total Edges: {total_edges}")
            dist = np.zeros(SF_NUM_AGENTS)
            color = "orange"
        else:
            return None, None
        
        total_final_ginis = []
        total_rounds = [i for i in range(total_edges)]
        # un-biased token to first agent
        # dist[0] = 1

        # un-biased token to random agent
        # dist[random.randint(0,len(dist)-1)] = 1

        # biased token to highest-degree agent
        # highest_node = max(G.degree, key=lambda x: x[1])[0]
        # dist[highest_node] = 1

        # biased token to lowest-degree agent?
        lowest_node = min(G.degree, key=lambda x: x[1])[0]
        dist[lowest_node] = 1
        
        # biased according to being in the Strongest Connected Component?
        # biased according to neighborhood connectedness?

        final_ginis = []
        for i in range(total_edges):
            edge_to_remove = random.choice(list(G.edges()))
            # apparently * will unpack into u, v for me
            G.remove_edge(*edge_to_remove)
            final_gini, ginis = run_simulation(G, dist)
            final_ginis.append(final_gini)
        total_final_ginis.append(final_ginis)
        axes[0].scatter([i/total_edges for i in range(0,total_edges)], final_ginis, alpha=0.3, s=30, color=color)

        column_means_of_ginis = np.mean(total_final_ginis, axis=0)
        normalized_rounds = [i/total_edges for i in range(0, total_edges)]
        x = np.asarray(total_rounds)
        x = np.asarray(normalized_rounds)
        y = np.asarray(column_means_of_ginis)

        # --- data-driven initial guesses ---
        A0 = float(np.percentile(y, 5))        # near lower asymptote
        K0 = float(np.percentile(y, 95))       # near upper asymptote
        x0_0 = float(np.median(x))             # center of x-range
        # finite-diff slope near middle to guess sign of k
        mid = int(len(x)/2)
        if 1 <= mid < len(x)-1:
            slope_mid = (y[mid+1] - y[mid-1]) / (x[mid+1] - x[mid-1])
        else:
            slope_mid = (y[-1] - y[0]) / max(1, (x[-1]-x[0]))
        # scale k by dynamic range and width; keep modest magnitude
        k0 = np.clip(4 * slope_mid / max(1e-6, (K0 - A0)), -5, 5)

        p0 = [A0, K0, k0, x0_0]

        # --- sensible bounds from data ---
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        pad_y = max(1e-3, 0.05*(ymax - ymin))  # small slack

        lb = [ymin - 5*pad_y, ymin,      -10, xmin]
        ub = [ymax + 5*pad_y, ymax + pad_y, 10,  xmax]

        FourParams, _ = curve_fit(logistic4, x, y, p0=p0, bounds=(lb, ub), maxfev=20000)

        A, K, k, x0 = FourParams
        print(f"{graph_type} INFLECTION: {x0}")
        total_params.append(FourParams)

    return total_params, total_edges

fig, axes = plt.subplots(1,2, figsize=(10,8))

comp_total_params, initial_comp_edges = gatherParams("COMP")
A_COMP, K_COMP, k_COMP, x0_COMP = np.mean(comp_total_params, axis=0)
xx = np.linspace(0, 1, 500)
yyFourParams = logistic4(xx, *np.mean(comp_total_params, axis=0))
axes[0].plot(xx, yyFourParams, label='COMP SIGMOID', linewidth=2, color='lightgreen')

y0_COMP = logistic4(x0_COMP, A_COMP, K_COMP, k_COMP, x0_COMP)
axes[0].scatter(
    [x0_COMP],            # x coordinate as a list or array
    [y0_COMP],            # y coordinate as a list or array
    color='black',
    s=80,
    zorder=5
)

rand_total_params, initial_rand_edges = gatherParams("RANDOM")
A_RAND, K_RAND, k_RAND, x0_RAND = np.mean(rand_total_params, axis=0)
xx = np.linspace(0, 1, 500)
yyFourParams = logistic4(xx, *np.mean(rand_total_params, axis=0))
axes[0].plot(xx, yyFourParams, label='RAND SIGMOID', linewidth=2, color='violet')

y0_RAND = logistic4(x0_RAND, A_RAND, K_RAND, k_RAND, x0_RAND)
axes[0].scatter(
    [x0_RAND],            # x coordinate as a list or array
    [y0_RAND],            # y coordinate as a list or array
    color='black',
    s=80,
    zorder=5
)

sw_total_params, initial_sw_edges = gatherParams("SW")
A_SW, K_SW, k_SW, x0_SW = np.mean(sw_total_params, axis=0)
xx = np.linspace(0, 1, 500)
yyFourParams = logistic4(xx, *np.mean(sw_total_params, axis=0))
axes[0].plot(xx, yyFourParams, label='SW SIGMOID', linewidth=2, color='lightblue')

y0_SW = logistic4(x0_SW, A_SW, K_SW, k_SW, x0_SW)
axes[0].scatter(
    [x0_SW],            # x coordinate as a list or array
    [y0_SW],            # y coordinate as a list or array
    color='black',
    s=80,
    zorder=5
)

sf_total_params, initial_sf_edges = gatherParams("SF")
A_SF, K_SF, k_SF, x0_SF = np.mean(sf_total_params, axis=0)
xx = np.linspace(0, 1, 500)
yyFourParams = logistic4(xx, *np.mean(sf_total_params, axis=0))
axes[0].plot(xx, yyFourParams, label='SF SIGMOID', linewidth=2, color='moccasin')

y0_SF = logistic4(x0_SF, A_SF, K_SF, k_SF, x0_SF)
axes[0].scatter(
    [x0_SF],            # x coordinate as a list or array
    [y0_SF],            # y coordinate as a list or array
    color='black',
    s=80,
    zorder=5
)

axes[0].legend()

# cf = Critical Fraction of Edges Deleted
cf_comp = x0_COMP
cf_rand = x0_RAND
cf_sw = x0_SW
cf_sf = x0_SF

# Stats
np_comp_total_params = np.array(comp_total_params)
x0_values = np_comp_total_params[:, 3]/initial_comp_edges
x0_q1 = np.percentile(x0_values, 25)
x0_q3 = np.percentile(x0_values, 75)
comp_x0_iqr = x0_q3 - x0_q1
print(f" COMP IQR    = {comp_x0_iqr:.3f} (Q1={x0_q1:.3f}, Q3={x0_q3:.3f})")

np_rand_total_params = np.array(rand_total_params)
x0_values = np_rand_total_params[:, 3]/initial_rand_edges
x0_q1 = np.percentile(x0_values, 25)
x0_q3 = np.percentile(x0_values, 75)
rand_x0_iqr = x0_q3 - x0_q1
print(f" RAND IQR    = {rand_x0_iqr:.3f} (Q1={x0_q1:.3f}, Q3={x0_q3:.3f})")

np_sw_total_params = np.array(sw_total_params)
x0_values = np_sw_total_params[:, 3]/initial_sw_edges
x0_q1 = np.percentile(x0_values, 25)
x0_q3 = np.percentile(x0_values, 75)
sw_x0_iqr = x0_q3 - x0_q1
print(f" SW IQR    = {sw_x0_iqr:.3f} (Q1={x0_q1:.3f}, Q3={x0_q3:.3f})")

np_sf_total_params = np.array(sf_total_params)
x0_values = np_sf_total_params[:, 3]
x0_q1 = np.percentile(x0_values, 25)
x0_q3 = np.percentile(x0_values, 75)
sf_x0_iqr = x0_q3 - x0_q1
print(f" SF IQR    = {sf_x0_iqr:.3f} (Q1={x0_q1:.3f}, Q3={x0_q3:.3f})")

print("Critical Percentage of Edges Deleted/Initial Edges for Sigmoid Inflection Point")
print("-------------------------------------------------------------------------------")
print(f"Complete: {np.round(cf_comp*100, 2)}% +- ({np.round(comp_x0_iqr*100,2)}%) Deleted")
print(f"Random: {np.round(cf_rand*100, 2)}% +- ({np.round(rand_x0_iqr*100,2)}%) Deleted")
print(f"Small-world: {np.round(cf_sw*100, 2)}% +- ({np.round(sw_x0_iqr*100,2)}%) Deleted")
print(f"Scale-free: {np.round(cf_sf*100, 2)}% +- ({np.round(sf_x0_iqr*100,2)}%) Deleted")
print("-------------------------------------------------------------------------------")

categories = ["RAND", "SW", "SF", "COMP"]
axes[1].barh(categories, [cf_rand, cf_sw, cf_sf, cf_comp], color="lightblue")
axes[0].set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axes[0].set_xlabel("Fraction of Edges Deleted")
axes[0].set_ylabel("Gini Coefficient")
axes[0].set_title("Final Gini After Edge Deletions")
axes[1].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axes[1].set_xlabel("Fraction of Edges Deleted From Initial Edges")
axes[1].set_title("Critical Fraction of Deletions for Sigmoid Inflection Point Per Network Type")

plt.tight_layout()
plt.show()

"""
UNNORMALIZED FIRST NODE
Critical Percentage of Edges Deleted/Initial Edges for Sigmoid Inflection Point
-------------------------------------------------------------------------------
Complete: 90.96% +- (1.65%) Deleted
Random: 71.74% +- (4.69%) Deleted
Small-world: 67.56% +- (4.41%) Deleted
Scale-free: 75.06% +- (4.66%) Deleted
-------------------------------------------------------------------------------

UNNORMALIZED RANDOM NODE
Critical Percentage of Edges Deleted/Initial Edges for Sigmoid Inflection Point
-------------------------------------------------------------------------------
Complete: 91.18% +- (3.69%) Deleted
Random: 81.32% +- (4.15%) Deleted
Small-world: 69.68% +- (3.38%) Deleted
Scale-free: 68.42% +- (21.45%) Deleted
-------------------------------------------------------------------------------

(Should I get rid of large outliers? Or are they important?)

NORMALIZED RANDOM NODE
Critical Percentage of Edges Deleted/Initial Edges for Sigmoid Inflection Point
-------------------------------------------------------------------------------
Complete: 95.42% +- (0.0%) Deleted
Random: 78.15% +- (0.0%) Deleted
Small-world: 69.5% +- (0.01%) Deleted
Scale-free: 69.16% +- (6.08%) Deleted
-------------------------------------------------------------------------------

NORMALIZED HIGHEST DEGREE NODE
Critical Percentage of Edges Deleted/Initial Edges for Sigmoid Inflection Point
-------------------------------------------------------------------------------
Complete: 94.77% +- (0.0%) Deleted
Random: 82.72% +- (0.0%) Deleted
Small-world: 71.19% +- (0.0%) Deleted
Scale-free: 75.17% +- (1.22%) Deleted
-------------------------------------------------------------------------------
"""