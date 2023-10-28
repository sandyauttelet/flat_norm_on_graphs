import numpy as np
import cubepy as cp
from time import time
import bst
from new_node import newnode
import pickle

#I haven't updated this yet.
filename = "2d_lookup_tree5k.txt"
c3_lookup_table = bst.load_tree(filename)

def integrand_3d(x,ui,uj):
    """call with cp.integrate(integrand,0,2*np.pi,([ui],[uj]))"""
    nu = np.zeros((len(x),3))[...,np.newaxis]
    nu[:,0] = np.sin(x[0])*np.cos(x[1])
    nu[:,1] = np.sin(x[0])*np.sin(x[1])
    nu[:,2] = np.cos(x[0])
    return np.abs(np.dot(ui,nu)*np.dot(uj,nu))

def c1_3d(u):
    """u should be a list of vectors [[u1],[u2],...[uD]]"""
    return np.pi*np.linalg.norm(u,axis=1)**2*(4/3)

def c2_3d(u):
    """u should be a list of vectors [[u1],[u2],...[uD]]"""
    return 2*np.pi*np.linalg.norm(u,axis=1)

def c3_3d_integral(u,k,m):
    k_norm = np.linalg.norm(u[k])
    m_norm = np.linalg.norm(u[m])
    k_hat = u[k]/k_norm
    m_hat = u[m]/m_norm
    theta_km = np.arccos(np.inner(k_hat,m_hat))
    result = k_norm*m_norm*bst.closest_angle(c3_lookup_table, theta_km)[1]
    return result
    #result = cp.integrate(integrand_2d, 0, 2*np.pi, ([u[k]],[u[m]]),itermax=25)
    #return result[0]
 
#This function was my solution for c3. I didn't figure out
#How to get both angles for the look up table like yours above.
def cube_C3(vector_pair,num):
    def sphere_integrand_explicit(theta,phi):
        return np.sin(theta)*(np.abs(np.sin(theta)*np.cos(phi)*vector_pair[0][0]\
            +np.sin(theta)*np.sin(phi)*vector_pair[0][1]+np.cos(theta)*vector_pair[0][2])\
            *np.abs(np.sin(theta)*np.cos(phi)*vector_pair[1][0]\
            +np.sin(theta)*np.sin(phi)*vector_pair[1][1]+np.cos(theta)*vector_pair[1][2]))
    low = [0.0,0.0]
    high = [np.pi,2*np.pi]
    value, error = cp.integrate(sphere_integrand_explicit, low, high,itermax=num)
    return value, error
    
def construct_weights_system_3d(u):
    col1 = c1_3d(u)
    b = c2_3d(u)
    D = len(u)
    lookup_table = {}
    A = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            if i == j:
                A[i,j] = col1[i]
            else:
                A[i,j] = c3_3d_integral(u,i,j)
    return A,b

def solve_weights_system_3d(u):
    A,b = construct_weights_system_3d(u)
    res = np.linalg.lstsq(A,b,rcond=None)[0]
    return res 

def get_num_integral_3d(u,k,m):
    return cp.integrate(integrand_3d, 0, 2*np.pi,([u[k]],[u[m]]),itermax=50)[0]
