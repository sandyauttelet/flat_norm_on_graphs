# -*- coding: utf-8 -*-
"""
Created on Mon May 29 07:34:37 2023

@author: Curtis
"""

import numpy as np

a = 0
b = 2*np.pi
num = 10000
dtheta = (b-a)/num
ui = np.array([3,9])
len_ui = np.linalg.norm(ui)

domain = np.linspace(a,b,num)
nus = [np.array([np.cos(theta),np.sin(theta)]) for theta in domain]
integrand = [np.inner(ui,nu)**2 for nu in nus]
integral = sum(integrand)*dtheta

print(integral)
print(np.pi*len_ui**2)