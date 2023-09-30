import sys
sys.path.append('../src/')
import system_eq_generator as sg
from sympy import *
import numpy as np
import functional as fu

#np.random.seed(0)

def generate_u(n,d=2,low=-10**3,high=10**3):
    """generate_u(n,d=2,low=-10,high=10)
    create a list of vectors representing edges to a vertex
    Parameters:
        n: int
            number of edges
        d: int dimension of vectors
        low: int lower bound of randint
        high: int upper bound of randint"""
    u = np.random.uniform(low,high,size=(n,d))
    return u

def generate_symbols(u):
    n = len(u)
    m = (n*n-n)//2
    c1 = symbols("1c:"+str(n))
    c2 = symbols("2c:"+str(n))
    c3 = symbols("3c:"+str(m))
    w = symbols("w:"+str(n))
    dictionary = {}
    index = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if str(i)+str(j) not in dictionary and str(j)+str(i) not in dictionary:
                    dictionary[str(i)+str(j)]=index
                    dictionary[str(j)+str(i)]=index
                    index += 1
    return c1,c2,c3,w,dictionary

def generate_system(c1,c2,c3,w,d):
    n = len(c1)
    f = lambda x: d[x]
    equations = []
    for i in range(n):
        exp = 2*w[i]*c1[i]-2*c2[i]
        for j in range(n):
            if i != j:
                exp += 2*w[j]*c3[f(str(i)+str(j))]
        equations.append(exp)
    return equations

def test_system(n):
    u = generate_u(n)
    #print(u)
    c1,c2,c3,w,dictionary = generate_symbols(u)
    #sys = generate_system(c1,c2,c3,w,dictionary)
    sg.numerical_vs_exact(u,None,w,dictionary)
    
def test_np(u):
    fu.solve_weights_system(u)
    
n = 100
for i in range(100):
    print(i)
    u = generate_u(n)
    test_np(u)

# for i in range(100):
#     print(i)
#     try:
#         test_system(10)
#     except IndexError:
#         pass
#     except Exception as err:
#         print(f"Unexpected {err=}, {type(err)=}")
#         raise

