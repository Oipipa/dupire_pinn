import torch
from ..losses.pde import dupire_residual
from ..autodiff import grad_wrt
from ..coords import K_from_x

def rmse_market(model,market):
    Cm,_,_=model(torch.stack([market.x,market.T],-1))
    return torch.sqrt(torch.mean((Cm-market.C).pow(2)))

def violations(model,xg,Tg,eps,S0,r,q):
    res,der=dupire_residual(model,xg,Tg,S0,r,q)
    v1=(der["CK"]>eps).sum()
    v2=(der["CKK"]<-eps).sum()
    v3=(der["CT"]<-eps).sum()
    return int((v1+v2+v3).item())

def norm_residual(model,xg,Tg,S0,r,q):
    res,der=dupire_residual(model,xg,Tg,S0,r,q)
    return torch.mean(res.abs()/(1+der["C"].abs()))

def mass_error(model,xline,Tline,S0,r,q):
    errs=[]
    for j in range(Tline.numel()):
        x=xline
        T=Tline[j].expand_as(x)
        xT=torch.stack([x,T],-1)
        xT.requires_grad_(True)
        C,_,_=model(xT)
        Cx=grad_wrt(C,xT,0)
        Cxx=torch.autograd.grad(Cx,xT,torch.ones_like(Cx),create_graph=False)[0][...,0]
        K=K_from_x(x,S0)
        CK=Cx/K
        CKK=(Cxx-Cx)/K.pow(2)
        lhs=torch.exp(q*Tline[j])*torch.trapz(CKK,K)
        rhs=torch.exp(-r*Tline[j])
        errs.append((lhs-rhs).abs())
    return torch.max(torch.stack(errs))
