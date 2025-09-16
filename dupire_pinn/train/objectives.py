import torch
from ..losses.pde import dupire_residual
from ..losses.noarb import no_arb_hinge
from ..losses.bc import bc_losses
from ..losses.regularizers import h1_nu,tv_sigma
from ..losses.huber import huber
from ..losses.tie import sigma_dupire_consistency

def objective(model,S0,r,q,weights,ints,bd,market=None,delta=1e-3,alpha_T=1.0,tv_eps=1e-6,arb_mode="hinge",sigma_ref=None,lam_pos=1e-3,lam_sigma=1e-4,lam_tie=1e-2):
    x=ints.x
    T=ints.T
    res,derivs=dupire_residual(model,x,T,S0,r,q)
    lpde=(res/S0).pow(2).mean()
    l0,linf,lT0=bc_losses(model,S0,r,q,bd.x0T,bd.xinfT,bd.xT0)
    lbc=(l0+linf+lT0)/S0**2
    lreg1=h1_nu(derivs["nu"],derivs["xT"],alpha_T)
    lreg2=tv_sigma(derivs["sigma"],derivs["xT"],tv_eps)
    larb=torch.tensor(0.0,device=x.device)
    if arb_mode=="hinge":
        larb=no_arb_hinge(derivs)
    lpos=((derivs["C"]/S0).neg().relu().pow(2).mean())
    lsig=torch.tensor(0.0,device=x.device)
    if sigma_ref is not None:
        lsig=(derivs["sigma"]-sigma_ref).pow(2).mean()
    ltie=sigma_dupire_consistency(derivs["C"],derivs["sigma"],derivs["xT"],S0,r,q,eps=1e-10,clip_max=3.0)
    lmkt=torch.tensor(0.0,device=x.device)
    if market is not None:
        xm=market.x; Tm=market.T
        Cm,_,_=model(torch.stack([xm,Tm],-1))
        lmkt=huber((Cm-market.C)/S0,delta).mean()
    L=weights["pde"]*lpde+weights["mkt"]*lmkt+weights["arb"]*larb+weights["bc"]*lbc+weights["reg1"]*lreg1+weights["reg2"]*lreg2+lam_pos*lpos+lam_sigma*lsig+lam_tie*ltie
    return L,{"pde":lpde,"mkt":lmkt,"arb":larb,"bc":lbc,"reg1":lreg1,"reg2":lreg2}
