import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import warnings

from time import perf_counter

# points_x = np.linspace(-2,2,30)
# points_y = np.linspace(-2,2,30)

# points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))

# points_disk = np.linalg.norm(points,axis=1)<=1

# plt.scatter(points[:,0],points[:,1])
# plt.scatter(points[points_disk][:,0],points[points_disk][:,1])

# t1 = perf_counter()

# =============================================================================
# above this is temporary testing stuff only
# =============================================================================

filename = '2d_lookup_table5000.txt'

file = np.loadtxt(filename,delimiter=',')

angles,values = file[:,0],file[:,1]

def get_perimeter(E,adjacency_matrix,points):
    #measure function
    s = 0.0
    n = len(points)
    for i in range(n):
        for j in range(n):
            if E[i] and not E[j]:
                s += adjacency_matrix[i,j]
    return s

def weights(i,j,u,u_lengths):
    #weight_function
    length = u_lengths[i]*u_lengths[j]
    if np.isclose(length,0):
        raise Exception("Point vector detected (length = 0)")
    if i == j:
        return np.pi*length
    inner = np.clip(np.inner(u[i],u[j])/length,-1.0,1.0)
    theta = np.arccos(inner)
    result = length*values[np.searchsorted(angles,theta).clip(0,len(values)-1)]
    return result

weights_vectorized = np.vectorize(weights,excluded=['u','u_lengths'])
#weight_function

solve_vectorized = np.vectorize(np.linalg.lstsq,excluded=['rcond'],signature='(m,m),(m)->(m),(k),(),(m)')
#weight_function

def make_b(lengths):
    #weight_function
    return 4*lengths

def make_A(edges,lengths):
    #weight_function
    n = len(edges)
    i,j = np.indices((n,n))
    A = weights_vectorized(i,j,u=edges,u_lengths=lengths)
    return A

A_vectorized = np.vectorize(make_A, signature='(m,n),(m)->(m,m)')
#weight_function

def get_weights(edges,lengths):
    #weight_function
    A = A_vectorized(edges,lengths)
    b = make_b(lengths)
    weights = solve_vectorized(A,b,rcond=None)[0]
    return weights

def get_sample(x_range,y_range,N):
    #voronoi function
    sample_points_x = np.random.uniform(*x_range,N)
    sample_points_y = np.random.uniform(*y_range,N)
    return np.column_stack((sample_points_x,sample_points_y))

def get_bounding_box(points):
    #voronoi function
    x_min,x_max = np.min(points[:,0]),np.max(points[:,0])
    y_min,y_max  = np.min(points[:,1]),np.max(points[:,1])
    x_range,y_range = (x_min,x_max),(y_min,y_max)
    area = np.linalg.norm(np.diff(x_range))*np.linalg.norm(np.diff(y_range))
    return x_range, y_range, area

def voronoi_areas(points,Tree,N=1000000):
    #voronoi function
    x_range, y_range, total_area = get_bounding_box(points)
    sample_points = get_sample(x_range, y_range, N)
    nearest_neighbors = Tree.query(sample_points)[1]
    indices = np.zeros(len(points))
    unique,counts = np.unique(nearest_neighbors,return_counts=True)
    indices[unique] = counts
    areas = total_area/N*indices
    return areas

def calculate_tree_graph(points,neighbors=24):
    #main function
    Tree = KDTree(points)
    graph = np.array(Tree.query(points,neighbors+1))
    neighbor_indices = graph[1,:,1:]
    n = len(points)
    i,j = np.indices((n,neighbors))
    weight_indices = np.dstack((i,neighbor_indices)).reshape(-1,2).astype(np.int32)
    return Tree,graph,weight_indices

def calculate_edge_vectors(points,graph):
    #main function
    lengths = graph[0,:,1:] #trim off the first column (all 0s)
    vertices = points[graph[1,:,1:].astype(np.int32)] #ditto as above
    edges = vertices - points[:,np.newaxis] #subtract to get vectors
    return edges,lengths,vertices

def add_source_sink(G,E,lamb):
    source = "source"
    sink = "sink"
    G.add_node(source)
    G.add_node(sink)
    for i,point in enumerate(E):
        if point:
            G.add_edge(source,i,weight=lamb)
        else:
            G.add_edge(sink,i,weight=lamb)

def flat_norm(points,E,lamb=1.0,perim_only=False,neighbors = 24):
    #main function
    n = len(points)
    if neighbors >= len(points):
        raise Exception("Need more points than neighbors")
    Tree,graph,weight_indices = calculate_tree_graph(points,neighbors)
    edges,lengths,vertices = calculate_edge_vectors(points,graph)
    areas = voronoi_areas(points,Tree)
    weights = get_weights(edges, lengths)
    scaled_weights = np.multiply(weights,areas[:,np.newaxis]).flatten()
    adjacency_matrix = np.zeros((n,n))
    for i,index in enumerate(weight_indices):
        adjacency_matrix[index[0]][index[1]] = scaled_weights[i]
    adjacency_matrix += adjacency_matrix.T
    P = get_perimeter(E, adjacency_matrix, points)
    if perim_only:
        return None,None,None,P
    G = nx.Graph(adjacency_matrix)
    add_source_sink(G,E,lamb)
    cut_value, partition = nx.minimum_cut(G,"source","sink",capacity='weight',flow_func=edmonds_karp)
    keep,_ = partition
    if len(keep) == 1:
        warnings.warn("No solution returned from min cut, lambda parameter likely too small.")
    keep.remove("source")
    result = points[list(keep)]
    #plt.scatter(result[:,0],result[:,1])
    sigma = np.zeros(n).astype(bool)
    sigma[list(keep)] = 1
    sigma = sigma.astype(bool)
    #print(sigma.shape)
    sigmac = ~sigma
    return result, sigma, sigmac, P

# =============================================================================
# below this is temporary testing stuff only
# =============================================================================

#flat_norm(points,points_disk,lamb=1e-2,neighbors=8)

# print(perf_counter() - t1)

# =============================================================================
# validating Kevin's weights
# =============================================================================

if __name__ == "__main__":
    u = np.array([(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
                  ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
                      ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
                          ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])

    u_lengths = np.linalg.norm(u,axis=1)

    result = get_weights(u,u_lengths)
    print("Ours: ", [result[0],result[5],result[9]])
    print("KRV: ", [0.1221, 0.0476, 0.0454])
