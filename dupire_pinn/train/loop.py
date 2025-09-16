import torch
import torch.nn.functional as F
from types import SimpleNamespace
from ..data.market import Market
from ..train.objectives import objective
from ..train.optimizer import make_optim
from ..losses.mass import mass_misfit
from ..autodiff import grad_wrt, hess_xx
from ..coords import K_from_x

def _constraints(model,S0,x,T):
    xT=torch.stack([x,T],-1)
    xT.requires_grad_(True)
    C,_,_=model(xT)
    Cx=grad_wrt(C,xT,0)
    Cxx=hess_xx(C,xT,0)
    CT=grad_wrt(C,xT,1)
    K=K_from_x(x,S0)
    CK=Cx/(K+1e-12)
    CKK=(Cxx-Cx)/((K+1e-12)**2)
    return CK,CKK,CT

def train(model,S0,r,q,ints,bd,market,weights,epochs=300,lr=5e-4,wd=0.0,delta=1e-3,alpha_T=1.0,tv_eps=1e-6,
          al=False,eta=0.5,sigma_ref=0.25,lam_pos=5e-3,lam_sigma=1e-3,lam_tie=1e-3,
          batch_int=4096,batch_mkt=2048,log_every=25,w_mass=0.0,xline_mass=None,T_mass=None):
    opt=make_optim(model.parameters(),lr,wd)
    N=ints.x.numel()
    if al:
        lam1=torch.zeros(N,dtype=torch.float64)
        lam2=torch.zeros(N,dtype=torch.float64)
        lam3=torch.zeros(N,dtype=torch.float64)
    for ep in range(epochs):
        perm=torch.randperm(N)
        run=0.0; steps=0
        for i in range(0,N,batch_int):
            idx=perm[i:i+batch_int]
            ib=SimpleNamespace(x=ints.x[idx],T=ints.T[idx])
            mb=None
            if market is not None:
                mN=market.x.numel()
                midx=torch.randint(0,mN,(min(batch_mkt,mN),))
                mb=Market(market.x[midx],market.T[midx],market.C[midx])
            opt.zero_grad()
            L,_=objective(model,S0,r,q,weights,ib,bd,mb,delta,alpha_T,tv_eps,
                          arb_mode="hinge",sigma_ref=sigma_ref,lam_pos=lam_pos,lam_sigma=lam_sigma,lam_tie=lam_tie)
            if al:
                CK,CKK,CT=_constraints(model,S0,ib.x,ib.T)
                la1=lam1[idx]; la2=lam2[idx]; la3=lam3[idx]
                A = (la1*CK + 0.5*eta*F.relu(CK).pow(2)) \
                    + (-la2*CKK + 0.5*eta*F.relu(-CKK).pow(2)) \
                    + (-la3*CT + 0.5*eta*F.relu(-CT).pow(2))
                L = L + weights["arb"]*A.mean()
            if w_mass>0.0 and xline_mass is not None and T_mass is not None:
                L=L+w_mass*mass_misfit(model,S0,r,q,xline_mass,T_mass)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            if al:
                with torch.no_grad():
                    CK,CKK,CT=_constraints(model,S0,ib.x,ib.T)
                    lam1[idx]=torch.clamp(lam1[idx]+eta*CK, min=0.0)
                    lam2[idx]=torch.clamp(lam2[idx]+eta*(-CKK), min=0.0)
                    lam3[idx]=torch.clamp(lam3[idx]+eta*(-CT), min=0.0)
            run+=float(L.detach()); steps+=1
        if (ep+1)%log_every==0:
            print(f"epoch {ep+1}/{epochs} loss {run/max(1,steps):.6f}")
    return model
