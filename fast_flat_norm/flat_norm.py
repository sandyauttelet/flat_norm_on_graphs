import pstats

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import warnings
from scipy.sparse import csr_array
from functools import wraps
from time import perf_counter
import cProfile
from numba import jit, int32, float64, types
import math

#number of physical cores
workers = 4

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print ('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap


points_x = np.linspace(-2,2,1000)
points_y = np.linspace(-2,2,1000)

# points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))

# points_disk = np.linalg.norm(points,axis=1)<=1

# plt.scatter(points[:,0],points[:,1])
# plt.scatter(points[points_disk][:,0],points[points_disk][:,1])

# t1 = perf_counter()

# =============================================================================
# above this is temporary testing stuff only
# =============================================================================

filename = '2d_lookup_table100000.txt'

file = np.loadtxt(filename,delimiter=',')

angles,values = file[:,0],file[:,1]

@jit(nopython=True)
def bs(angles,theta):
    left,right = 0, len(angles)-1
    eps = 1e-8
    while (left <= right):
        mid = (left+ right)//2

        if abs(angles[mid] - theta) < eps:
            return mid
        elif angles[mid] < theta:
            left = mid + 1
        else:
            right = mid - 1
    return right+1

@timing
def get_perimeter(E,G):
    #measure function
    s = 0.0
    for point in G.nodes:
        if E[point]:
            point_edges = G.edges(point)
            for edge in point_edges:
                p1,p2 = edge
                if not E[p2]:
                    s += G[p1][p2]["weight"]
    return s

@jit(nopython=True)
def weights_numba(i,j,u,u_lengths,values):
    #weight_function
    length = u_lengths[i]*u_lengths[j]
    if i == j:
        return math.pi*length
    # try to jiggle into -1,1
    eps = 10e-6
    inner = sum([u[i][k]*u[j][k] for k in range(len(u[i]))])
    inner = inner/(length+eps)
    theta = math.acos(inner)
    idx = bs(angles,theta)#np.searchsorted(angles,theta)
    # idx.clip(0,len(values)-1)
    result = length*values[idx]
    return result

weights_numba = np.vectorize(weights_numba,excluded=['u','u_lengths','values'])
#weight_function

#solve_vectorized = np.vectorize(np.linalg.lstsq,excluded=['rcond'],signature='(m,m),(m)->(m),(k),(),(m)')
#weight_function

@jit(types.Array(float64,1,"C")(types.Array(float64,2,"C"),types.Array(float64,1,"C")),nopython=True)
def fast_lst_sqs(A,b):
    lstsq_soln = np.linalg.lstsq(A,b)
    sing_vals = lstsq_soln[3]
    #print(np.max(sing_vals)/np.min(sing_vals))
    return lstsq_soln[0]

@timing
@jit(types.Array(float64,2,"C")(types.Array(float64,3,"C"),types.Array(float64,2,"C")),nopython=True)
def A_solver(A_vec,b_vec):
    n = len(A_vec)
    m = len(A_vec[0])
    soln_vec = np.zeros((n,m))
    for i in range(n):
        soln_vec[i] = fast_lst_sqs(A_vec[i],b_vec[i])
    return soln_vec

def make_b(lengths):
    #weight_function
    return 4*lengths

def make_A(edges,lengths):
    #weight_function
    n = len(edges)
    i,j = np.indices((n,n))
    A = weights_numba(i,j,u=edges,u_lengths=lengths,values=values)
    return A

A_vectorized = np.vectorize(make_A, signature='(m,n),(m)->(m,m)')
#weight_function

@timing
def get_weights(edges,lengths):
    A = A_vectorized(edges,lengths)
    b = make_b(lengths)
    weights = A_solver(A,b)
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

@timing
def voronoi_areas(points,Tree,N=1000000):
    #voronoi function
    x_range, y_range, total_area = get_bounding_box(points)
    sample_points = get_sample(x_range, y_range, N)
    nearest_neighbors = Tree.query(sample_points,workers=workers)[1]
    indices = np.zeros(len(points))
    unique,counts = np.unique(nearest_neighbors,return_counts=True)
    indices[unique] = counts
    areas = total_area/N*indices
    return areas

@timing
def calculate_tree_graph(points,neighbors=24):
    #main function
    Tree = KDTree(points)
    graph = np.array(Tree.query(points,neighbors+1,workers=workers))
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
    #print(edges[len(edges)//2+1])
    print("neighbor edges:", edges[len(edges)//2+2])
    print("vertex:", vertices[len(edges)//2+2])
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

@timing
def get_min_cut(G):
    return nx.minimum_cut(G,"source","sink",capacity='weight')

def flat_norm(points,E,lamb=1.0,perim_only=False,neighbors = 24):
    #main function
    n = len(points)
    if neighbors >= len(points):
        raise Exception("Need more points than neighbors")
    weightst0 = perf_counter()
    Tree,graph,weight_indices = calculate_tree_graph(points,neighbors)
    edges,lengths,vertices = calculate_edge_vectors(points,graph)
    areas = voronoi_areas(points,Tree)
    weights = get_weights(edges, lengths)
    scaled_weights = np.multiply(weights,areas[:,np.newaxis]).flatten()
    weightst1 = perf_counter()
    perimt0 = perf_counter()
    row = weight_indices[:, 0]
    col = weight_indices[:, 1]
    # this has the same effect as the adding entries for csr format
    weights = np.append(scaled_weights,scaled_weights)
    rows = np.append(row,col)
    cols = np.append(col,row)
    sparse = csr_array((weights, (rows, cols)), shape=(n, n))
    G = nx.from_scipy_sparse_array(sparse)
    perim = get_perimeter(E,G)
    perimt1 = perf_counter()



    if perim_only:
        return None,None,None,perim

    mft0 = perf_counter()
    add_source_sink(G,E,lamb)
    cut_value, partition = get_min_cut(G)
    keep,_ = partition
    mft1 = perf_counter()

    times = [mft1-mft0,weightst1 - weightst0,perimt1 - perimt0]
    total = sum(times)
    # print(f"Time to finish weights\n raw: {weightst1 - weightst0}\n %: {(weightst1 - weightst0) / total}")
    # print()
    # print(f"Time to finish perim\n raw: {perimt1 - perimt0}\n %: {(perimt1 - perimt0) / total}")
    # print()
    #print(f"Time to finish min cut\n raw: {mft1 - mft0}\n %: {(mft1 - mft0) / total}")
    if len(keep) in [0,1]:
        warnings.warn("No solution returned from min cut, lambda parameter likely too small.")
    if len(keep) >= 1:
        keep.remove("source")

    result = points[list(keep)]
    #plt.scatter(result[:,0],result[:,1])
    sigma = np.zeros(n).astype(bool)
    sigma[list(keep)] = 1
    sigma = sigma.astype(bool)
    #print(sigma.shape)
    sigmac = ~sigma
    return result, sigma, sigmac, perim

# =============================================================================
# below this is temporary testing stuff only
# =============================================================================

#flat_norm(points,points_disk,lamb=1e-2,neighbors=8)

# print(perf_counter() - t1)

# =============================================================================
# validating Kevin's weights
# =============================================================================

if __name__ == "__main__":
    # u = np.array([(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
    #               ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
    #                   ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
    #                       ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])
    points = np.array([(0.0,0.0),(-1.0,0.0),(0.0, -1.0)\
                  ,(-1.0,1.0),(1.0,1.0)\
                      ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
                          ])
    
    # points_x = np.arange(-25, 25, 1)
    # points_y = np.arange(-25, 25, 1)
    # points = np.dstack(np.meshgrid(points_x,points_y)).reshape(-1,2)
    #u_lengths = np.linalg.norm(u,axis=1)
    flat_norm(points,np.ones(len(points)),lamb=1.0,neighbors=8)
    #result = get_weights(u,u_lengths)
    #print("Ours: ", [result[0],result[5],result[9]])
    print("KRV: ", [0.1221, 0.0476, 0.0454])

    # tick = perf_counter()
    # flat_norm(points, points_disk, lamb=1, neighbors=8)
    # tock = perf_counter()
    # print(tock-tick)
    import pstats
    from pstats import SortKey

# =============================================================================
#     points_x = np.linspace(-2, 2, 100)
#     points_y = np.linspace(-2, 2, 100)
#     points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
# 
#     points_disk = np.linalg.norm(points,axis=1)<=1
#     flat_norm(points, points_disk, lamb=1.0, neighbors=24)
# =============================================================================
    #cProfile.run('flat_norm(points, points_disk, lamb=.001, neighbors=8)',"flatnorm")
    #p = pstats.Stats("flatnorm")
    #p.strip_dirs().sort_stats(SortKey.TIME).print_stats(50)