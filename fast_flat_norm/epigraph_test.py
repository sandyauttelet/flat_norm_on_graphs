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

def epigraph(f,grid_points,xmin=-1.0,xmax=1.0,ymin=-1.0,ymax=1.0):
    points_x = np.linspace(xmin,xmax,grid_points)
    points_y = np.linspace(ymin,ymax,grid_points)
    box_points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
    truth_array = []
    for point in box_points:
        if f(point[0]) <= point[1]:
            truth_array.append(True)
        else:
            truth_array.append(False)
    return box_points,np.array(truth_array)

def test(f,df,grid_points=30,neighbors=8):
    points,E = epigraph(f,grid_points)
    plt.scatter(points[E][:,0],points[E][:,1])
    #plt.plot(points[:,0],f(points[:,1]))
    #plt.scatter(points[~E][:,0],points[~E][:,1])
    arclengthint = arc_length(-1,1,df)
    print("arc length numerically:", arclengthint)
    t1 = perf_counter()
    fn_est = flat_norm(points,E,neighbors=neighbors,perim_only=True)
    print("flat norm est: ", fn_est)
    print("Rel err: ", np.abs((arclengthint-fn_est)/arclengthint)*100.0)
    print("time to run:",perf_counter() - t1 )
    plt.show()

if __name__ == "__main__":
    freq = 1
    neighbors = 16
    grid_points = 200
    test(*(make_sin_freq(freq)),grid_points=grid_points,neighbors=neighbors)
    """
    freq = 1
    neighbors = 16
    grid_points = 200
    arc length numerically: 12.056910087083894
    flat norm est:  8.44974750791295
    Rel err:  29.917802763040918
    """
    #test(f,df,100)


