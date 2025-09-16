import torch
from ..autodiff import grad_wrt, hess_xx
from ..coords import K_from_x

def mass_misfit(model, S0, r, q, xline, Tvals):
    losses=[]
    for T in Tvals:
        xT=torch.stack([xline, torch.full_like(xline, T)], -1)
        xT.requires_grad_(True)
        C,_,_=model(xT)
        Cx=grad_wrt(C,xT,0)
        Cxx=hess_xx(C,xT,0)
        K=K_from_x(xline,S0)
        CKK=(Cxx-Cx)/(K**2)
        val=torch.exp(q*T)*torch.trapz(CKK,K)-torch.exp(-r*T)
        losses.append(val.pow(2))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=xline.device, dtype=xline.dtype)
