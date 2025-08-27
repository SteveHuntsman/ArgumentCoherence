
# %% Test graph
import networkx as nx
from copy import deepcopy

E = [('a', 'e', 2),
     ('a', 'k', 2),
     ('a', 'p', 1),
     ('a', 'r', 1),
     ('e', 'k', 2),
     ('e', 'p', -2),
     ('e', 'r', -2),
     ('k', 'p', -2),
     ('k', 'r', 1),
     ('o', 'p', 2),
     ('o', 'r', 2),
     ('p', 'r', 2)
     ]

G = nx.Graph()  
G.add_weighted_edges_from(E)

G_neg = deepcopy(G) 
nx.set_edge_attributes(G_neg, {e: -G.edges[e]['weight'] for e in G.edges}, 'weight')

# %%
import networkx as nx
import itertools
from collections import defaultdict

def max_cut_local_search(G, seed_bipartition=None, max_moves=2, return_cut_value=True, max_solutions=10, return_full_output=False):
    """
    A MAX-CUT solver that explores bipartitions within a certain number of 
    element exchanges from a seed bipartition and returns all optimal solutions up to max_solutions.
    
    Parameters:
    -----------
    G : networkx.Graph
        An undirected graph with weighted edges
    seed_bipartition : list, optional
        Initial bipartition (subset of nodes). If None, an empty set is used.
    max_moves : int
        Maximum number of element moves to consider
    return_cut_value : bool
        If True, return the cut value along with each bipartition.
        If False, only return the bipartitions.
    max_solutions : int, default=10
        Maximum number of optimal solutions to return
    
    Returns:
    --------
    If return_cut_value is True:
        list of tuples : [(list, list, float), ...]
            The bipartitions of nodes and their maximum cut values
    If return_cut_value is False:
        list of tuples : [(list, list), ...]
            Just the bipartitions of nodes
    If return_full_output is True:
        verbose : [(list, list, float), ...]
            The bipartitions of nodes and their cut values
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Initialize seed bipartition if not provided
    if seed_bipartition is None:
        seed_bipartition = []
    
    # Convert to sets for efficient operations
    part_1 = set(seed_bipartition)
    part_0 = set(nodes) - part_1
    
    # Function to calculate cut value between two parts
    def calculate_cut_value(p0, p1):
        cut = 0
        for i in p0:
            for j in p1:
                if G.has_edge(i, j):
                    cut += G[i][j].get('weight', 1)
        return cut
    
    # Store cut values and corresponding bipartitions
    cut_to_bipartitions = defaultdict(list)
    
    # Calculate initial cut value
    initial_cut_value = calculate_cut_value(part_0, part_1)
    cut_to_bipartitions[initial_cut_value].append((list(part_0), list(part_1)))
    
    if return_full_output:
        verbose = []

    # For each number of moves from 1 to max_moves
    for k in range(1, max_moves + 1):
        # Generate all possible k-combinations from each part
        for p0_elements in itertools.combinations(part_0, min(k, len(part_0))):
            for p1_elements in itertools.combinations(part_1, min(k, len(part_1))):
                
                # Create new bipartitions by exchanging elements
                new_p0 = (part_0 - set(p0_elements)) | set(p1_elements)
                new_p1 = (part_1 - set(p1_elements)) | set(p0_elements)
                
                # Calculate new cut value
                cut_value = calculate_cut_value(new_p0, new_p1)
                
                # Store the bipartition with its cut value
                # Create a canonical representation to avoid duplicates
                # Sort both parts and use a frozenset to ensure uniqueness
                canonical_bipartition = (
                    frozenset(sorted(new_p0)),
                    frozenset(sorted(new_p1))
                )

                if return_full_output:
                    verbose.append([list(new_p0), list(new_p1), cut_value])
                
                # Add to the dictionary only if we haven't seen this bipartition before
                is_new = True
                for existing_bipartition in cut_to_bipartitions[cut_value]:
                    existing_p0, existing_p1 = existing_bipartition
                    if (frozenset(existing_p0) == canonical_bipartition[0] and 
                        frozenset(existing_p1) == canonical_bipartition[1]) or \
                       (frozenset(existing_p0) == canonical_bipartition[1] and 
                        frozenset(existing_p1) == canonical_bipartition[0]):
                        is_new = False
                        break
                
                if is_new:
                    cut_to_bipartitions[cut_value].append((list(new_p0), list(new_p1)))
    
    # Find the maximum cut value
    max_cut = max(cut_to_bipartitions.keys())
    
    # Get all bipartitions with the maximum cut value (up to max_solutions)
    best_bipartitions = cut_to_bipartitions[max_cut][:max_solutions]
    
    if return_cut_value:
        # Return bipartitions with their cut value
        if return_full_output:
            return [(bp[0], bp[1], max_cut) for bp in best_bipartitions], verbose
        else:
            return [(bp[0], bp[1], max_cut) for bp in best_bipartitions]
    else:
        # Return just the bipartitions
        if return_full_output:
            return best_bipartitions, verbose
        else:
            return best_bipartitions

# %% https://claude.site/artifacts/f4cc0b2b-8814-4963-8fb1-549797e3479b
# Cf. https://doi.org/10.1007/978-0-387-30162-4_219
import networkx as nx
import numpy as np
from scipy.optimize import linprog
import itertools

def max_cut_linear_relaxation(G):
    """
    Solve the linear relaxation of the MAX-CUT problem on a weighted graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        A weighted undirected graph. Edges should have a 'weight' attribute.
        If no weight is specified, a default weight of 1 is assumed.
    
    Returns:
    --------
    dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    float
        The optimal objective value.
    """
    # Get list of nodes and edges
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create a mapping from node to index
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # Initialize the objective function coefficients (all zeros initially)
    c = np.zeros((n * (n-1)) // 2)
    
    # Create a mapping from edge to its position in the flattened variable array
    # Create variables for ALL pairs of vertices, not just existing edges
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx  # Both directions map to the same variable
            idx += 1
    
    # Populate the objective function coefficients
    # For pairs that aren't edges in G, w_ij is set to 0 (as per the problem statement)
    for i in range(n):
        for j in range(i+1, n):
            u, v = idx_to_node[i], idx_to_node[j]
            idx = edge_to_idx[(i, j)]
            
            # Check if edge exists and get its weight
            if G.has_edge(u, v):
                weight = G[u][v].get('weight', 1.0)  # Default weight is 1.0
            else:
                weight = 0.0  # Non-edges have weight 0
                
            # Maximize sum(w_ij * x_ij), or minimize sum(-w_ij * x_ij)
            c[idx] = -weight
    
    # Create constraints
    A_ub = []
    b_ub = []
    
    # For each 3-cycle i,j,k:
    # 1. x_ij + x_jk + x_ki <= 2
    # 2. x_ij + x_jk - x_ki >= 0 => -x_ij - x_jk + x_ki <= 0
    for i, j, k in itertools.combinations(range(n), 3):
        # First constraint: x_ij + x_jk + x_ki <= 2
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(i, j)]] = 1
        constraint[edge_to_idx[(j, k)]] = 1
        constraint[edge_to_idx[(k, i)]] = 1
        A_ub.append(constraint)
        b_ub.append(2)
        
        # Second constraint: x_ij + x_jk - x_ki >= 0 => -x_ij - x_jk + x_ki <= 0
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(i, j)]] = -1
        constraint[edge_to_idx[(j, k)]] = -1
        constraint[edge_to_idx[(k, i)]] = 1
        A_ub.append(constraint)
        b_ub.append(0)
        
        # Cyclic permutations for the second constraint
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(j, k)]] = -1
        constraint[edge_to_idx[(k, i)]] = -1
        constraint[edge_to_idx[(i, j)]] = 1
        A_ub.append(constraint)
        b_ub.append(0)
        
        constraint = np.zeros(len(c))
        constraint[edge_to_idx[(k, i)]] = -1
        constraint[edge_to_idx[(i, j)]] = -1
        constraint[edge_to_idx[(j, k)]] = 1
        A_ub.append(constraint)
        b_ub.append(0)
    
    # Linear relaxation: 0 <= x_ij <= 1 (bounds in linprog)
    bounds = [(0, 1) for _ in range(len(c))]
    
    # Solve the linear relaxation
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Check if the optimization was successful
    if result.success:
        # Convert the solution back to a dictionary mapping edges to values
        solution = {}

        idx_to_edge = {idx: (min(i, j), max(i, j)) for (i, j), idx in edge_to_idx.items() if i < j}
        for idx, val in enumerate(result.x):
            if idx in idx_to_edge:
                i, j = idx_to_edge[idx]
                u, v = nodes[i], nodes[j]
                # Only include variables for actual edges in the result
                # (non-edges have weight 0 and don't contribute to objective)
                if G.has_edge(u, v) or abs(val) > 1e-6:  # Include if significant value or is an edge
                    solution[(u, v)] = val

        # for i in range(n):
        #     for j in range(i+1, n):
        #         idx = edge_to_idx[(i, j)]
        #         u, v = idx_to_node[i], idx_to_node[j]
        #         solution[(u, v)] = result.x[idx]
        #         solution[(v, u)] = result.x[idx]  # Add both directions for convenience
        
        return solution, -result.fun, result.x, edge_to_idx, node_to_idx, idx_to_node  # Negate because we minimized -objective
    else:
        raise ValueError(f"Optimization failed: {result.message}")

# %%
import random

def construct_partition_deterministic(G, x_values, seed_node=None):
    """
    Construct a partition of the graph nodes based on the LP relaxation solution.
    This uses a deterministic approach starting from a seed node.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    x_values : dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    seed_node : any, optional
        The node to start the partition from. If None, the first node is used.
        
    Returns:
    --------
    tuple
        A tuple (S, V_S) representing the bipartition of the vertices.
    """
    nodes = list(G.nodes())
    if not nodes:
        return set(), set()
    
    if seed_node is None:
        seed_node = nodes[0]
    
    # Initialize the partition with the seed node
    S = {seed_node}
    V_S = set()
    
    # Assign remaining nodes based on their x_ij values relative to the seed node
    for node in nodes:
        if node == seed_node:
            continue
        
        # If x_seed,node is closer to 1, they should be on opposite sides
        if x_values.get((seed_node, node), 0) >= 0.5:
            V_S.add(node)
        else:
            S.add(node)
    
    return S, V_S

def construct_partition_randomized(G, x_values, seed=None):
    """
    Construct a partition of the graph nodes based on the LP relaxation solution.
    This uses a randomized approach.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    x_values : dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        A tuple (S, V_S) representing the bipartition of the vertices.
    """
    if seed is not None:
        random.seed(seed)
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Calculate the bias term epsilon for each node
    epsilon = {}
    for node in nodes:
        # Average tendency of this node to be on different side than others
        sum_x = sum(x_values.get((node, other), 0) for other in nodes if other != node)
        epsilon[node] = (sum_x / (n-1 if n > 1 else 1)) - 0.5
    
    # Assign nodes to partitions based on randomized rounding
    S = set()
    V_S = set()
    
    for node in nodes:
        # Probability of putting node in S: 0.5 - epsilon
        # (If epsilon > 0, node tends to be on different sides with others, so less likely in S)
        if random.random() >= 0.5 + epsilon[node]:
            S.add(node)
        else:
            V_S.add(node)
    
    return S, V_S

def construct_best_partition(G, x_values, num_trials=10):
    """
    Try multiple random partitions and return the one with highest cut weight.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    x_values : dict
        A dictionary mapping edge tuples (i,j) to their relaxed x_ij values.
    num_trials : int
        Number of random partitions to try.
        
    Returns:
    --------
    tuple
        A tuple (S, V_S) representing the best bipartition of the vertices.
    float
        The cut weight of the best partition.
    """
    best_partition = None
    best_weight = -float('inf')
    
    # Try the deterministic partition
    S, V_S = construct_partition_deterministic(G, x_values)
    weight = calculate_cut_weight(G, S, V_S)
    if weight > best_weight:
        best_weight = weight
        best_partition = (S, V_S)
    
    # Try multiple random partitions
    for i in range(num_trials):
        S, V_S = construct_partition_randomized(G, x_values, seed=i)
        weight = calculate_cut_weight(G, S, V_S)
        if weight > best_weight:
            best_weight = weight
            best_partition = (S, V_S)
    
    return best_partition, best_weight

def calculate_cut_weight(G, S, V_S):
    """
    Calculate the weight of the cut defined by the partition (S, V_S).
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    S : set
        First set of the partition.
    V_S : set
        Second set of the partition.
        
    Returns:
    --------
    float
        The total weight of edges crossing the cut.
    """
    cut_weight = 0.0
    
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        # If the edge crosses the cut (one endpoint in S, the other in V_S)
        if (u in S and v in V_S) or (u in V_S and v in S):
            cut_weight += weight
    
    return cut_weight

# %% Adapted from https://blog.lalovic.io/max-cut-sdp/ under https://creativecommons.org/licenses/by/4.0/deed.en
# https://claude.ai/share/bb78b187-d6b2-407d-8153-020530cf81b2
import numpy as np
import cvxpy as cp
import networkx as nx
from scipy.linalg import sqrtm

def gw_weighted(G):

    """
    Goemans-Williamson algorithm for weighted MaxCut:
    Given a weighted graph G, returns a vector x in {-1, 1}^n
    that corresponds to the chosen subset of vertices S of V.
    
    Parameters:
    -----------
    G : networkx.Graph
        Weighted undirected graph. Edge weights should be stored in the 'weight' attribute.
    
    Returns:
    --------
    x : numpy.ndarray
        Vector in {-1, 1}^n representing the partition of vertices.
    """
    n = G.number_of_nodes()
    
    # Get mapping from node names to indices (if nodes are not 0...n-1)
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    
    # Get edges and weights
    weighted_edges = []
    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        # Default weight to 1 if not specified
        weight = data.get('weight', 1.0)
        weighted_edges.append((i, j, weight))
    
    ## SDP Relaxation
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]  # X is positive semidefinite
    constraints += [
        X[i, i] == 1 for i in range(n)  # Diagonal entries are 1
    ]
    
    # Weighted objective function
    objective = sum(0.5 * weight * (1 - X[i, j]) for (i, j, weight) in weighted_edges)
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()
    
    ## Hyperplane Rounding
    Q = sqrtm(X.value).real
    r = np.random.randn(n)
    x = np.sign(Q @ r)
    
    return x

def weighted_cut(x, G):
    """
    Given a vector x in {-1, 1}^n and a weighted graph G,
    returns the edges in cut(S) and the total weight of the cut
    for the subset of vertices S of V represented by x.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Vector in {-1, 1}^n representing the partition of vertices.
    G : networkx.Graph
        Weighted undirected graph.
        
    Returns:
    --------
    cut_edges : list
        List of edges in the cut.
    cut_weight : float
        Total weight of the cut.
    """

    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    cut_edges = []
    cut_weight = 0.0
    
    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = data.get('weight', 1.0)
        
        if np.sign(x[i] * x[j]) < 0:
            cut_edges.append((u, v))
            cut_weight += weight
            
    return cut_edges, cut_weight

# %%
if __name__ == "__main__":
    # Solve the LP relaxation
    x_values, obj_value, res_x, edge_to_idx, node_to_idx, idx_to_node = max_cut_linear_relaxation(G_neg)
    
    print(f"Relaxed solution objective value: {obj_value:.4f}")
    print("\nRelaxed x_ij values:")
    for edge, value in sorted(x_values.items()):
        if edge[0] < edge[1]:  # Only print each edge once
            print(f"x_{edge} = {value:.4f}")

    # Generate partitions
    S_det, V_S_det = construct_partition_deterministic(G_neg, x_values)
    weight_det = calculate_cut_weight(G_neg, S_det, V_S_det)
    print(f"\nDeterministic partition: S = {S_det}, V\\S = {V_S_det}")
    print(f"Cut weight: {weight_det}")
    
    S_rand, V_S_rand = construct_partition_randomized(G_neg, x_values, seed=42)
    weight_rand = calculate_cut_weight(G_neg, S_rand, V_S_rand)
    print(f"\nRandomized partition: S = {S_rand}, V\\S = {V_S_rand}")
    print(f"Cut weight: {weight_rand}")
    
    best_partition, best_weight = construct_best_partition(G_neg, x_values, num_trials=1000)
    print(f"\nBest partition: S = {best_partition[0]}, V\\S = {best_partition[1]}")
    print(f"Best cut weight: {best_weight}")
    
    # Calculate approximation ratio
    approx_ratio = best_weight / obj_value if obj_value != 0 else float('inf')
    print(f"\nApproximation ratio: {approx_ratio:.4f}")

    # Run Goemans-Williamson algorithm ONE TIME
    num_trials = 100
    best_gw = -np.inf
    for i in range(num_trials):
        x = gw_weighted(G_neg)
    
        # Get the cut
        cut_edges, cut_weight = weighted_cut(x, G_neg)

        # 
        if cut_weight > best_gw:
            part0 = [list(G_neg.nodes())[i] for i in np.where(x == -1.)[0]]
            part1 = [list(G_neg.nodes())[i] for i in np.where(x == 1.)[0]]
            best_gw_cut = cut_edges
            best_gw_weight = cut_weight

    print(f"\nSDP partition: S = {part1}, V\\S = {part0}")
    print(f"Cut weight: {best_gw_weight}")
    print(f"Cut edges: {best_gw_cut}")