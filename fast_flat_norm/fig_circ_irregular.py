import numpy as np
import matplotlib.pyplot as plt 
import sys
from flat_norm import flat_norm
from perturb_points import perturb_points

n = 100
x = np.linspace(-2,2,n)
y = np.linspace(-2,2,n)

points = np.dstack(np.meshgrid(x,y)).reshape(-1,2)

#points = perturb_points(points,1.0)

plt.scatter(points[:,0],points[:,1],label="Grid")

center1 = np.array([-0.5,-0.5])
r1 = 0.5
cond1 = np.linalg.norm(points-center1,axis=1)<=r1
center2 = np.array([0.5,0.5])
r2 = 0.25
cond2 = np.linalg.norm(points-center2,axis=1)<=r2
circ = cond1 + cond2

circ_pts = (points[circ][:,0],points[circ][:,1])
plt.scatter(*circ_pts,label="$\Omega$")

neighbors = 24
lamb=0.0078125/2
Omega = circ
fn_est,sigma,sigmac,perim = flat_norm(points,Omega,lamb,perim_only=False,neighbors=neighbors)
plt.scatter(points[sigma][:,0],points[sigma][:,1],color='black',label="$\Sigma$")

plt.legend()

