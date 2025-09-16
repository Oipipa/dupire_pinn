import torch
import torch.nn as nn
from .mlp import mlp

class SharedTwoHead(nn.Module):
    def __init__(self,in_dim=2,width=256,hidden=4):
        super().__init__()
        self.trunk=mlp(in_dim,hidden,width,width)
        self.head_C=nn.Linear(width,1)
        self.head_nu=nn.Linear(width,1)
    def forward(self,xT):
        h=self.trunk(xT)
        C=self.head_C(h).squeeze(-1)
        nu=self.head_nu(h).squeeze(-1)
        sigma=torch.exp(0.5*nu)
        return C,nu,sigma
