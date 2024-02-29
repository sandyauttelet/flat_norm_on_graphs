from flat_norm import flat_norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
from time import perf_counter

def f(x):
    return x**2

def df(x):
    return 2*x

def make_sin_freq(a):
    def h(x):
        return np.sin(2*np.pi*a*x)
    def dh(x):
        return 2*np.pi*a*np.cos(2*np.pi*a*x)
    return h,dh

def arc_length(a,b,df):
    return fixed_quad(lambda x: np.sqrt(1+df(x)**2),a,b)[0]

def epigraph(f,xmin=-1.0,xmax=1.0,ymin=-1.0,ymax=1.0):
    points_x = np.linspace(xmin,xmax,30)
    points_y = np.linspace(ymin,ymax,30)
    box_points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
    truth_array = []
    for point in box_points:
        if f(point[0]) <= point[1]:
            truth_array.append(True)
        else:
            truth_array.append(False)
    return box_points,np.array(truth_array)

def test(f,df,neighbors=8):
    points,E = epigraph(f)
    plt.scatter(points[E][:,0],points[E][:,1])
    #plt.scatter(points[~E][:,0],points[~E][:,1])
    print("arc length numerically:", arc_length(-1,1,df))
    t1 = perf_counter()
    print("flat norm value: ", end="")
    flat_norm(points,E,neighbors)
    print("time to run:",perf_counter() - t1 )
    
freq = 2
neighbors = 8
test(*(make_sin_freq(freq)),neighbors)
#test(f,df)


