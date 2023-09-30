# -*- coding: utf-8 -*-
"""
Created on Mon May 29 07:34:37 2023

@author: Curtis
"""

import numpy as np

a = 0
b = 2*np.pi
num = 100000
dtheta = (b-a)/num
ui = np.array([1,2])
len_ui = np.linalg.norm(ui)

domain = np.linspace(a,b,num)
nus = [np.array([np.cos(theta),np.sin(theta)]) for theta in domain]
integrand = [ np.abs(np.inner(ui,nu)) for nu in nus]
integral = sum(integrand)*dtheta

print(integral)
print(4*len_ui)