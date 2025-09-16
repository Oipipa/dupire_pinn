import torch
import torch.nn as nn
from .heads import SharedTwoHead
from ..utils.bs import bs_call
from ..coords import K_from_x

class BaselineCall(nn.Module):
    def __init__(self,width=128,hidden=4,bs0=0.25,S0=1.0,r=0.0,q=0.0):
        super().__init__()
        self.inner=SharedTwoHead(in_dim=2,width=width,hidden=hidden)
        self.bs0=bs0; self.S0=S0; self.r=r; self.q=q
        self.softplus=nn.Softplus()
    def forward(self,xT):
        Craw,nu,sigma=self.inner(xT)
        x=xT[...,0]; T=xT[...,1]
        K=K_from_x(x,self.S0)
        Cb=bs_call(self.S0,K,T,self.r,self.q,torch.full_like(T,self.bs0))
        C=self.softplus(Craw)+Cb
        return C,nu,sigma
