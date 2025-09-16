import torch

def grad_vec(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]

def grad_wrt(y,x,idx):
    g=grad_vec(y,x)
    return g[...,idx]

def hess_xx(y,x,idx=0):
    gy=grad_wrt(y,x,idx)
    g2=torch.autograd.grad(gy,x,torch.ones_like(gy),create_graph=True)[0]
    return g2[...,idx]
