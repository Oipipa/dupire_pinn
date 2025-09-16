import torch
from ..autodiff import grad_wrt

def h1_nu(nu,xT,alpha_T=1.0):
    gx=grad_wrt(nu,xT,0)
    gT=grad_wrt(nu,xT,1)
    return (gx.pow(2)+alpha_T*gT.pow(2)).mean()

def tv_sigma(sigma,xT,eps=1e-6):
    gx=grad_wrt(sigma,xT,0)
    gT=grad_wrt(sigma,xT,1)
    return (gx.pow(2)+gT.pow(2)+eps*eps).sqrt().mean()
