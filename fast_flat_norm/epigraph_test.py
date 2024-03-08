from flat_norm import flat_norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from time import perf_counter
from perturb_points import perturb_points

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
    return quad(lambda x: np.sqrt(1+df(x)**2),a,b)[0]

def epigraph(f,grid_points,xmin=-1.0,xmax=1.0,ymin=-1.0,ymax=1.0,perturb=False):
    points_x = np.linspace(xmin,xmax,grid_points)
    points_y = np.linspace(ymin,ymax,grid_points)
    box_points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
    if perturb:
        box_points = perturb_points(box_points)
    truth_array = []
    for point in box_points:
        if f(point[0]) <= point[1]:
            truth_array.append(True)
        else:
            truth_array.append(False)
    return box_points,np.array(truth_array)

def test(f,df,grid_points=30,neighbors=8,perturb=False):
    if perturb:
        points,E = epigraph(f,grid_points,perturb=True)
    else:
        points,E = epigraph(f,grid_points)
    plt.scatter(points[E][:,0],points[E][:,1],color='mediumblue',label="$\Sigma \cap E$")
    #plt.plot(points[:,0],f(points[:,1]))
    plt.scatter(points[~E][:,0],points[~E][:,1],color='dimgray',label="$\Sigma^c \cap E^c$")
    arclengthint = arc_length(-1,1,df)
    print("arc length numerically:", arclengthint)
    t1 = perf_counter()
    fn_est,sigma,sigmac,perim = flat_norm(points,E,neighbors=neighbors,lamb=0.005)
    print("flat norm est: ", perim)
    print("Rel err: ", np.abs((arclengthint - perim) / arclengthint) * 100.0)
    print("time to run:", perf_counter() - t1)
    set1 = sigmac & E
    set2 = sigma & ~E
    plt.scatter(points[set1][:,0],points[set1][:,1],color='indianred',label="$\Sigma^c \cap E$")
    plt.scatter(points[set2][:,0],points[set2][:,1],color='cornflowerblue', label="$\Sigma \cap E^c$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        #plt.figure()
        freq = 1
        neighbors = 8
        grid_points = 50
        tick = perf_counter()
        test(*(make_sin_freq(freq)),grid_points=grid_points,neighbors=neighbors,perturb=True)
        tock = perf_counter()
        print(tock-tick)
    """
    freq = 1
    neighbors = 16
    grid_points = 200
    arc length numerically: 12.056910087083894
    flat norm est:  8.44974750791295
    Rel err:  29.917802763040918
    """
    #test(f,df,100)


