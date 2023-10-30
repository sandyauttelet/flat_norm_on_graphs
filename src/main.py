import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import functional as f
import optimizer as opt
import graph as g  
import image_to_graph as im2g
from new_node import newnode

import cProfile

img_path = '20x20.png'

# def flatnorm(nodes,A,l):
#     edges = g.make_complete(nodes)
#     G = g.euclidean_graph(nodes,edges)
#     full_vector_list = []
#     for node in nodes:
#         full_vector_list += list(g.get_edge_vectors(G, node)[0])
#     norms = [np.linalg.norm(vector) for vector in full_vector_list]
#     print("minimum norm: ", min(norms))
#     opt.get_graph_weights(G)
#     g.min_cut_max_flow(G,A,l)

# def flatnorm_circle(r,n,A,l):
#     nodes = [(r*np.cos(x),r*np.sin(x)) for x in np.linspace(0,2*np.pi - .5,n)] #points on unit circle
#     flatnorm(nodes,A,l)

# def flatnorm_grid(n,A,l):
#     nodes = [(i,j) for i in range(-n,n) for j in range(-n,n)] #grid
#     flatnorm(nodes,A,l)

from time import perf_counter

def timer(fn):
    def wrapper(*args,**kwargs):
        t1 = perf_counter()
        result = fn(*args,**kwargs)
        t2 = perf_counter()
        print(f"Time to run: {t2-t1} seconds or {(t2-t1)/60} minutes")
        return result
    return wrapper

@timer
def main():
    graph,in_set = im2g.image_to_graph(img_path)

    edges = g.make_complete(graph)
    G = g.euclidean_graph(graph,edges)
    reduced = g.knn_reduced_graph(G, 24)
    opt.get_graph_weights(reduced)
    t1 = perf_counter()
    reduced = g.min_cut_max_flow(reduced,in_set,30)
    t2 = perf_counter()
    print(f"Min cut max flow took {t2-t1} seconds to run")
    xs,ys=[],[]
    try:
        reduced.remove("source")
    except None as e:
        pass
    for x,y in reduced:
        xs.append(x)
        ys.append(y)
    plt.figure()
    plt.scatter(xs, ys)
    
cProfile.run('main()', 'results')

import pstats

with open('results.txt', 'w') as file:
    profile = pstats.Stats('results', stream=file)
    profile.print_stats()
    file.close()

