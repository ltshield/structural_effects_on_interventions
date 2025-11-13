import random
from typing import List, Tuple, Dict, Any

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Globals
SEED = 42
NUM_AGENTS = 40
NUM_ROUNDS = 100
RUNS_PER_DELETION = 100          # repeat per deletion count to average/IQR
DELETE_BIAS = "uniform"          # "uniform" or "degree"

random.seed(SEED)
np.random.seed(SEED)

FRAC_OF_TOTAL_EDGES = 0.25
APPROX_NUM_EDGES = (NUM_AGENTS)*(NUM_AGENTS-1)/2

# Gini Coefficient Calculator
def gini(x: np.ndarray) -> float:
    """
    Gini coefficient for nonnegative vector x (not all zeros).
    Mass scale-invariant; works with any total.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    # Using the well-known formula: G = (sum_i (2i-n-1) x_i) / (n * sum x_i)
    index = np.arange(1, n + 1)
    return (np.dot((2 * index - n - 1), xs)) / (n * s)

# Row-stochastic Adjacency Matrix
def build_transition_matrix(G: nx.Graph) -> np.ndarray:
    """
    Row-stochastic transition matrix P where each node splits its tokens
    equally among its neighbors. If a node is isolated, it keeps its tokens (self-loop).
    """
    n = G.number_of_nodes()
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        nbrs = list(G.neighbors(i))
        if len(nbrs) == 0:
            P[i, i] = 1.0
        else:
            w = 1.0 / len(nbrs)
            for j in nbrs:
                P[i, j] = w
    return P

# Token Exchange Behavior
def simulate_token_exchange(
    G: nx.Graph,
    initial: np.ndarray,
    num_rounds: int = NUM_ROUNDS
) -> Tuple[float, List[float]]:
    """
    Closed-world token exchange: x_{t+1} = x_t @ P
    Returns (final_gini, ginis_per_round)
    """
    P = build_transition_matrix(G)
    x = initial.copy().astype(float)
    ginis = []
    for _ in range(num_rounds):
        # compute gini for *current* round distribution
        ginis.append(gini(x))
        x = x @ P
    final_g = gini(x)
    return final_g, ginis

# Edge Deletion Logic
def pick_edges_to_delete(G: nx.Graph, k: int, mode: str = "uniform") -> List[Tuple[int, int]]:
    """
    Choose k distinct edges to delete from G (without replacement).
    mode:
      - 'uniform': all existing edges equally likely
      - 'degree': probability proportional to sum of endpoint degrees (bias toward high-degree edges)
    """
    edges = list(G.edges())
    if k > len(edges):
        k = len(edges)
    if mode == "uniform":
        return random.sample(edges, k)
    elif mode == "degree":
        # weight each edge by deg(u)+deg(v)
        deg = dict(G.degree())
        weights = np.array([deg[u] + deg[v] for u, v in edges], dtype=float)
        wsum = weights.sum()
        if wsum <= 0:
            return random.sample(edges, k)
        probs = weights / wsum
        # sample without replacement using probabilities
        # NumPy choice with replace=False supports p
        idxs = np.random.choice(len(edges), size=k, replace=False, p=probs)
        return [edges[i] for i in idxs]
    else:
        raise ValueError("Unknown DELETE_BIAS mode")
    
# All Steps Tied Together
def run_experiment(
    num_agents: int = NUM_AGENTS,
    runs_per_deletion: int = RUNS_PER_DELETION,
    delete_bias: str = DELETE_BIAS
):
    # base complete graph
    base = nx.complete_graph(num_agents)
    total_edges = base.number_of_edges()

    # initial token distribution: one token at node 0
    initial = np.zeros(num_agents, dtype=float)
    initial[0] = 1.0

    # gather per-deletion results
    # deletion_results[d] -> list of dicts { 'final_gini': float, 'ginis': [..], 'edges_deleted': [...] }
    deletion_results: Dict[int, List[Dict[str, Any]]] = {}

    # We'll iterate d = 1..total_edges (each trial: rebuild complete, delete d edges, simulate)
    for d in range(1, total_edges + 1):
        deletion_results[d] = []
        for r in range(runs_per_deletion):
            G = nx.complete_graph(num_agents)
            to_del = pick_edges_to_delete(G, d, mode=delete_bias)
            G.remove_edges_from(to_del)

            final_g, ginis = simulate_token_exchange(G, initial, NUM_ROUNDS)

            deletion_results[d].append({
                "final_gini": float(final_g),
                "ginis": ginis,
                "edges_deleted": to_del
            })

    return deletion_results, total_edges

def IQR_PlotForlast60(deletion_results: dict, total_edges: int, window_size: int = int(FRAC_OF_TOTAL_EDGES*(APPROX_NUM_EDGES)/6)):
    """
    For the last 60 deletions (or fewer), group by 10 (e.g., 110–119, 120–129, ...),
    and, for each group, plot mean (across deletions) ± IQR of the per-round mean
    Gini trajectories (mean across runs for each deletion count).
    """

    # --- robust num_rounds discovery ---
    if not deletion_results:
        raise ValueError("deletion_results is empty.")
    # take any deletion count key, then any trial within it
    some_d = next(iter(deletion_results))
    if not deletion_results[some_d]:
        raise ValueError(f"No trials stored for deletion count {some_d}.")
    num_rounds = len(deletion_results[some_d][0]["ginis"])
    rounds = np.arange(num_rounds)

    # --- choose last 60 deletions (or less) ---
    last_n = min(int(FRAC_OF_TOTAL_EDGES*APPROX_NUM_EDGES), total_edges)
    start_d = max(1, total_edges - last_n + 1)
    relevant_ds = list(range(start_d, total_edges + 1))

    # group into windows of size 10
    groups = [relevant_ds[i:i + window_size] for i in range(0, len(relevant_ds), window_size)]

    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

    for gi, group_ds in enumerate(groups):
        # for each deletion count in this group, get the mean-per-round across runs
        mean_trajs = []
        for d in group_ds:
            runs = deletion_results.get(d, [])
            if not runs:
                continue
            run_trajs = np.array([trial["ginis"] for trial in runs], dtype=float)  # (runs, R)
            mean_traj = run_trajs.mean(axis=0)  # (R,)
            mean_trajs.append(mean_traj)

        if not mean_trajs:
            continue

        mean_trajs = np.vstack(mean_trajs)  # (num_deletions_in_group, R)

        q25 = np.percentile(mean_trajs, 25, axis=0)
        q75 = np.percentile(mean_trajs, 75, axis=0)
        mean_of_means = mean_trajs.mean(axis=0)

        label = f"Deletions {group_ds[0]}–{group_ds[-1]}"
        plt.plot(rounds, mean_of_means, color=colors[gi], label=label)
        plt.fill_between(rounds, q25, q75, color=colors[gi], alpha=0.25)

    plt.title("Gini over time — last 60 deletions in sets of 10 (mean ± IQR)")
    plt.xlabel("Round")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Deletion Ranges", fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    deletion_results, total_edges = run_experiment(
        num_agents=NUM_AGENTS,
        runs_per_deletion=RUNS_PER_DELETION,
        delete_bias=DELETE_BIAS
    )

    # Used to generate figure: "IterativeEdgeDeletion_Gini_IQRLast60.png"
    IQR_PlotForlast60(deletion_results, total_edges)