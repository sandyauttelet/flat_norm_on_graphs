# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:01:13 2023

@author: Curtis
"""

import torch

n = 1000

def f(nu):
    return torch.inner(nu,torch.tensor([0,1.0]))**2

def spherical_simpsons(f,n):
    domain = torch.linspace(0,2*torch.pi,n+1)
    x_component = torch.cos(domain)
    y_component = torch.sin(domain)
    nu = torch.stack([x_component,y_component],1)
    h = 2*torch.pi/n
    summands = [f(nu[i])+4*f((nu[i]+nu[i+1])/2)+\
                f(nu[i+1]) for i in range(n)]
    return h/6*sum(summands)

print(spherical_simpsons(f,n))