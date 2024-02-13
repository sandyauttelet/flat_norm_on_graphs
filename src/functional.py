import numpy as np
import cubepy as cp
from time import time
import bst
from new_node import newnode
import pickle

filename = "2d_lookup_tree5k.txt"
c3_lookup_table = bst.load_tree(filename)

def integrand_2d(x,ui,uj):
    """
        Finds 2d integrand for constant integration of C3.

        Parameters
        ----------
        x : float
            DESCRIPTION
        ui : array of floats
            specific vector used for integration.
        uj : array of floats
            vector which is connected to the specific vector used for integration.

        Returns
        -------
        float
            integrand used for integration of C3.

        """
    """call with cp.integrate(integrand,0,2*np.pi,([ui],[uj]))"""
    nu = np.zeros((len(x),2))[...,np.newaxis]
    nu[:,0] = np.cos(x)
    nu[:,1] = np.sin(x)
    return np.abs(np.dot(ui,nu)*np.dot(uj,nu))

def c1_2d(u):
    """u should be a list of vectors [[u1],[u2],...[uD]]"""
    return np.pi*np.linalg.norm(u,axis=1)**2

def c2_2d(u):
    """
        Finds constant values for C2 in 2d derived from spherical jacobian.

        Parameters
        ----------
        u : list of list of floats
            list of vectors created from data set.

        Returns
        -------
        float
            constant value for C2 in 2d.

        """
    """u should be a list of vectors [[u1],[u2],...[uD]]"""
    return 4*np.linalg.norm(u,axis=1)

def c3_2d_integral_test(u,k,m):
    """integrate using gauss kronrod through cubepy"""
    k_norm = np.linalg.norm(u[k])
    m_norm = np.linalg.norm(u[m])
    k_hat = u[k]/k_norm
    m_hat = u[m]/m_norm
    #print(k_hat,m_hat)
    theta_km = np.arccos(np.inner(k_hat,m_hat))
    result = bst.closest_angle(c3_lookup_table, theta_km)[1]
    angle_close = bst.closest_angle(c3_lookup_table, theta_km)[0]
    #print(k_norm,m_norm,theta_km,angle_close)
    #print(k_norm,m_norm)
    #print()
    #print(theta_km,angle_close)
    result *= k_norm*m_norm
    
    return result

def memoize(f):
    """maintains a cache"""
    cache = {}
    def wrap(*args,**kwargs):
        key = str(args) + str(kwargs)
        if key in cache:
            return cache[key]
        result = f(*args,**kwargs)
        cache[key] = result
        return result
    return wrap

@memoize
def c3_2d_integral(u,k,m):
    k_norm = np.linalg.norm(u[k])
    m_norm = np.linalg.norm(u[m])
    k_hat = u[k]/k_norm
    m_hat = u[m]/m_norm
    theta_km = np.arccos(np.inner(k_hat,m_hat))
    result = k_norm*m_norm*bst.closest_angle(c3_lookup_table, theta_km)[1]
    return result
    # result = cp.integrate(integrand_2d, 0, 2*np.pi, ([u[k]],[u[m]]),itermax=25)
    # return result[0]

def construct_weights_system(u):
    """
        Creates jacobian matrix for vectors and their solution to be used in minimization of weights.

        Parameters
        ----------
        u : list of list of floats
            list of vectors created from data set.

        Returns
        -------
        A : array of array of floats
            coefficient matrix for linear optimization problem.
        b : array of floats
            solution vector for linear optimization problem, a list of all C2 constants for each vector.

        """
    col1 = c1_2d(u)
    b = c2_2d(u)
    D = len(u)
    lookup_table = {}
    A = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            if i == j:
                A[i,j] = col1[i]
            else:
                A[i,j] = c3_2d_integral(u,i,j)
    return A,b

def solve_weights_system(u):
    """
        Solves the linear optimization problem to compute the minimized weights of the graph.

        Parameters
        ----------
        u : list of list of floats
            list of vectors created from data set.

        Returns
        -------
        res : list of floats
            least squares solution to the minimization problem, i.e. the optimal weights of our graph.

        """
    # t0 = time()
    A,b = construct_weights_system(u)
    # t1 = time()
    # result = np.linalg.solve(A,b)
    # t2 = time()
    # ax = b
    # x = a^-1b
    # pinv = np.linalg.pinv(A)
    # res = pinv @ b
    res = np.linalg.lstsq(A,b,rcond=None)[0]
    #At = A.T
    #res = np.linalg.solve(At@A,At@b)
    #print("Time to construct:", t1-t0, "Time to solve:", t2-t1 )
    return res   

def get_num_integral(u,k,m):
    return cp.integrate(integrand_2d, 0, 2*np.pi,([u[k]],[u[m]]),itermax=50)[0]

def test_unit_circle(r):
    p1 = [[r*np.cos(x),r*np.sin(x)] for x in np.linspace(0,2*np.pi,10)]
    errs = []
    for i,_ in enumerate(p1):
        for j,_ in enumerate(p1):
            if i!=j:
                table_res = c3_2d_integral(p1,i,j)
                num_res = get_num_integral(p1,i,j)
                errs.append(abs(table_res-num_res))
                #assert np.close(table_res,num_res)
    print(max(errs))

# 0.015155662532481351 5k
def test_vector_mag():
    u = np.array([[1.0,0.0],[-3.0,2.0]])
    errs = []
    for i in range(2,10):
        u *= i
        #print(u)
        table_res = c3_2d_integral(u,0,1)
        num_res = get_num_integral(u,0,1)
        errs.append(abs(table_res-num_res))
    print(max(errs))
    print(min(errs))
    #print(errs)
#test_unit_circle(18)
#test_vector_mag()

def compare_with_paper():
    u = np.array([(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
                  ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
                      ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
                          ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])
    result = solve_weights_system(u)
    print("Ours: ", [round(result[0],4),round(result[5],4),round(result[9],4)])
    print("KRV: ", [0.1221, 0.0476, 0.0454])
    return u,result
    
if __name__ == '__main__':
    u,result = compare_with_paper()
    from matplotlib.pyplot import quiver
    x = np.zeros(len(u))
    y = np.zeros(len(u))
    u = map(np.array,u)
    u = np.array(list(u))
    widths = [np.linalg.norm(entry) for entry in u]
    u,v = u[:,0],u[:,1]
    quiver(x,y,u,v,linewidths = widths)




    
    