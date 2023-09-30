import sys
sys.path.append('../src/')
import system_eq_generator as sg
from sympy import *
import numpy as np
import functional as fu
import numerical_code as num
import graph as g
import optimizer as opt
import torch
#define the graph

vertices = [torch.tensor([0.0,0.0]),torch.tensor([1.0,1.0]),torch.tensor([1.0,0.0]),torch.tensor([0.0,1.0])]

#list of edges for a complete graph
connections = [[torch.tensor([1.0,1.0])-torch.tensor([1.0,0.0]),\
               torch.tensor([1.0,1.0])-torch.tensor([0.0,1.0]),
               torch.tensor([1.0,1.0])-torch.tensor([0.0,0.0])]\
               ,[torch.tensor([0.0,1.0])-torch.tensor([1.0,0.0]),
              torch.tensor([0.0,1.0])-torch.tensor([0.0,0.0])],
               [torch.tensor([0.0,0.0])-torch.tensor([1.0,0.0])]]


def comp_linsolve_to_newtonsolve(nodes,plot=True):
    edges = g.make_complete(nodes)
    G = g.euclidean_graph(nodes,edges)
    opt.get_graph_weights(G)
    for node in G.nodes():
        us,edges = g.get_edge_vectors_for_numsolve(G, node)
        connections = [torch.tensor(u).float() for u in us]
        print("connections: ",connections)
        num_weights = num.get_node_weights(connections).detach()
        for i in range(0,len(edges),2):
            p1,p2 = edges[i]
            linsolve_weight = G[p1][p2]["functional_weight"]
            # adjust numerical code to compute weights in both directions 
            # and add them together
            num_weight = num_weights[i].item()+num_weights[i+1].item()
            print(linsolve_weight,num_weight)
            assert np.allclose(linsolve_weight,num_weight,atol=1e-2)
            
    if plot: g.euclidean_plot_2d(G)
 
# =============================================================================
# simple test on regular grid
# =============================================================================
nodes = [(0.0,0.0),(1.0,1.0),(1.0,0.0),(0.0,1.0)]

comp_linsolve_to_newtonsolve(nodes)
    

# =============================================================================
# vertices sampled from the unit circle                        
# =============================================================================

#nodes = [(np.cos(x),np.sin(x)) for x in np.arange(0,np.pi*2,1)]
#print(nodes)
#comp_linsolve_to_newtonsolve(nodes)
