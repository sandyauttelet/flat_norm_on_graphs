import numpy as np

from functional import solve_weights_system
from graph import get_edge_vectors, euclidean_graph, make_complete, euclidean_plot_2d

from tqdm import tqdm

def get_graph_weights(G):
    cap = "capacity"
    for node in tqdm(G.nodes):
        u,edges = get_edge_vectors(G,node)
        functional_weights = solve_weights_system(u)
        for i,u in enumerate(edges):
            p1,p2 = u
            if cap in G[p1][p2]: 
                G[p1][p2][cap] += functional_weights[i]
            else:
                G[p1][p2][cap] = functional_weights[i]
