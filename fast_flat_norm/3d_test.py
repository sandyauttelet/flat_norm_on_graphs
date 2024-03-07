"""
ID: bwukki1
LANG: PYTHON3
TASK: test
"""

import numpy as np
import matplotlib.pyplot as plt 
from sympy import *
from flat_norm_3d import flat_norm

points_x = np.linspace(-2,2,30)
points_y = np.linspace(-2,2,30)
points_z = np.linspace(-2,2,30)

xs,ys,zs = np.meshgrid(points_x,points_y,points_z)

x_flat = xs.flatten()
y_flat = ys.flatten()
z_flat = zs.flatten()

points = np.dstack((x_flat,y_flat,z_flat)).reshape(-1,3)

points_disk = np.linalg.norm(points,axis=1)<=1

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(points[:,0], points[:,1], points[:,2])
#ax.scatter(points[points_disk][:,0],points[points_disk][:,1],points[points_disk][:,2],c="red")


flat_norm(points,points_disk,8)