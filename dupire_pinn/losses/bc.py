import torch
from ..autodiff import grad_wrt
from ..coords import K_from_x,CK_from_Cx

def bc_losses(model,S0,r,q,x0T,xinfT,xT0):
    device=x0T.device if x0T.numel()>0 else (xinfT.device if xinfT.numel()>0 else xT0.device)
    l0=torch.tensor(0.0,device=device)
    linf=torch.tensor(0.0,device=device)
    lT0=torch.tensor(0.0,device=device)
    if xT0.numel()>0:
        xT0.requires_grad_(True)
        C0,_,_=model(xT0)
        K=K_from_x(xT0[...,0],S0)
        payoff=torch.clamp(S0-K,min=0.0)
        lT0=(C0-payoff).pow(2).mean()
    if x0T.numel()>0:
        x0T.requires_grad_(True)
        Cb,_,_=model(x0T)
        Cx=grad_wrt(Cb,x0T,0)
        T=x0T[...,1]
        K=K_from_x(x0T[...,0],S0)
        CK=CK_from_Cx(Cx,K)
        tgtC=S0*torch.exp(-q*T)
        tgtdC=-torch.exp(-r*T)
        l0=(Cb-tgtC).pow(2).mean()+(CK-tgtdC).pow(2).mean()
    if xinfT.numel()>0:
        xinfT.requires_grad_(True)
        Cw,_,_=model(xinfT)
        Cxw=grad_wrt(Cw,xinfT,0)
        Kw=K_from_x(xinfT[...,0],S0)
        CKw=CK_from_Cx(Cxw,Kw)
        linf=Cw.pow(2).mean()+CKw.pow(2).mean()
    return l0,linf,lT0
