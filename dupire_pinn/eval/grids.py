import torch
from ..coords import K_from_x

def grid(xmin,xmax,Tmin,Tmax,nx,nt,S0):
    x=torch.linspace(xmin,xmax,nx,dtype=torch.float64)
    T=torch.linspace(Tmin,Tmax,nt,dtype=torch.float64)
    X,Tm=torch.meshgrid(x,T,indexing="ij")
    K=K_from_x(X,S0)
    return X.reshape(-1),Tm.reshape(-1),K.reshape(-1),x,T
