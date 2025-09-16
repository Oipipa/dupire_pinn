import torch
from ..coords import K_from_x,CK_from_Cx,CKK_from_Cxx_Cx
from ..autodiff import grad_wrt,hess_xx

def sigma_dupire_consistency(C,sigma,xT,S0,r,q,eps=1e-10,clip_max=None):
    x=xT[...,0]; T=xT[...,1]
    Cx=grad_wrt(C,xT,0)
    CT=grad_wrt(C,xT,1)
    Cxx=hess_xx(C,xT,0)
    K=K_from_x(x,S0)
    CK=CK_from_Cx(Cx,K)
    CKK=CKK_from_Cxx_Cx(Cxx,Cx,K)
    num=CT+(r-q)*K*CK+q*C
    den=0.5*K.pow(2)*CKK
    mask=den.abs()>eps
    impl=torch.zeros_like(num)
    impl[mask]=torch.sqrt(torch.relu(num[mask]/den[mask]))
    if clip_max is not None:
        impl=torch.clamp(impl,max=clip_max)
    return (sigma-impl).pow(2)[mask].mean() if mask.any() else torch.tensor(0.0,device=x.device)
