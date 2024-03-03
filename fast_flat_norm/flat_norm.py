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

filename = '2d_lookup_table5000.txt'

file = np.loadtxt(filename,delimiter=',')

angles,values = file[:,0],file[:,1]

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

def weights(i,j,u,u_lengths):
    #weight_function
    length = u_lengths[i]*u_lengths[j]
    if i == j:
        return np.pi*length
    # try to jiggle into -1,1
    eps = np.finfo(np.float32).eps
    inner = np.inner(u[i],u[j])/(length+eps)
    theta = np.arccos(inner)
    idx = np.searchsorted(angles,theta)
    # idx.clip(0,len(values)-1)
    result = length*values[idx]
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

@timing
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

@timing
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

@timing
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

@timing
def get_min_cut(G):
    return nx.minimum_cut(G,"source","sink",capacity='weight',flow_func=edmonds_karp)

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
    #
    # u_lengths = np.linalg.norm(u,axis=1)
    #
    # result = get_weights(u,u_lengths)
    # print("Ours: ", [result[0],result[5],result[9]])
    # print("KRV: ", [0.1221, 0.0476, 0.0454])

    # tick = perf_counter()
    # flat_norm(points, points_disk, lamb=1, neighbors=8)
    # tock = perf_counter()
    # print(tock-tick)
    import pstats
    from pstats import SortKey
    cProfile.run('flat_norm(points, points_disk, lamb=1, neighbors=8)',"flatnorm")
    p = pstats.Stats("flatnorm")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(50)