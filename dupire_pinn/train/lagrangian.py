import torch
from ..losses.pde import dupire_residual

def init_multipliers(n,device):
    l1=torch.zeros(n,device=device,dtype=torch.float64)
    l2=torch.zeros(n,device=device,dtype=torch.float64)
    l3=torch.zeros(n,device=device,dtype=torch.float64)
    return l1,l2,l3

def al_term(model,x,T,S0,r,q,lambdas,eta,update=True):
    res,der=dupire_residual(model,x,T,S0,r,q)
    CK=der["CK"]
    CKK=der["CKK"]
    CT=der["CT"]
    l1,l2,l3=lambdas
    t=l1*CK - l2*CKK - l3*CT + 0.5*eta*(torch.relu(CK).pow(2)+torch.relu(-CKK).pow(2)+torch.relu(-CT).pow(2))
    L=t.mean()
    if update:
        l1.data=torch.clamp(l1+eta*CK,min=0)
        l2.data=torch.clamp(l2-eta*CKK,min=0)
        l3.data=torch.clamp(l3-eta*CT,min=0)
    return L,(l1,l2,l3)
