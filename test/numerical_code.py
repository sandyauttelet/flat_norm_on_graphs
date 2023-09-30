import torch
import matplotlib.pyplot as plt
from torch.func import hessian
from torch.func import grad


#n is the number of vectors on the sphere for integration

n = 180

#parameters for Newton's
maxit = 100
tol = 10**(-7)

def newton_direction(f,x_k,u):
    b = -grad(f)(x_k,u)
    A = hessian(f)(x_k,u)
    pinv = torch.linalg.pinv(A)
    #x = torch.linalg.solve(A,b)
    x = pinv @ b
    return x

def newton(f,x_0,u):
    x_prev = x_0
    for i in range(maxit):
        newton_dir = newton_direction(f,x_prev,u)
        x_prev = x_prev
        x_next = x_prev + newton_dir
        if torch.linalg.norm(x_prev-x_next) < tol:
            return x_next
        else:
            x_prev = x_next
    raise RuntimeError("Max iterations in Newton's Reached")

def spherical_simpsons(f,n,omegas,u):
    domain = torch.linspace(0,2*torch.pi,n+1)
    x_component = torch.cos(domain)
    y_component = torch.sin(domain)
    nu = torch.stack([x_component,y_component],1)
    h = 2*torch.pi/n
    summands = [f(nu[i],omegas,u)+4*f((nu[i]+nu[i+1])/2,omegas,u)+\
                f(nu[i+1],omegas,u) for i in range(n)]
    return h/6*sum(summands)

def integrand(nu,omegas,u):
    #print(u)
    summands = 0
    for i,omega in enumerate(omegas):
        summands += omega*torch.abs(torch.inner(nu,u[i]))
    result = (summands - 1)**2
    return result

def F(omegas,u):
    integral = spherical_simpsons(integrand,n,omegas,u)
    return integral

def get_weights(u):
    degree = len(u)
    x_0 = torch.zeros(degree)
    return newton(F,x_0,u)

def get_node_weights(connections):
    weights = get_weights(connections)
    return weights

u = [(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
              ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
                  ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
                      ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)]
    
u = [torch.tensor(entry) for entry in u]
    
print(get_node_weights(u))