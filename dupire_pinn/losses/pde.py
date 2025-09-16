import torch
from ..coords import CK_from_Cx,CKK_from_Cxx_Cx,K_from_x
from ..autodiff import grad_wrt,hess_xx

def dupire_residual(model,x,T,S0,r,q):
    xT=torch.stack([x,T],-1)
    xT.requires_grad_(True)
    C,nu,sigma=model(xT)
    Cx=grad_wrt(C,xT,0)
    CT=grad_wrt(C,xT,1)
    Cxx=hess_xx(C,xT,0)
    K=K_from_x(x,S0)
    CK=CK_from_Cx(Cx,K)
    CKK=CKK_from_Cxx_Cx(Cxx,Cx,K)
    res=CT-0.5*sigma.pow(2)*K.pow(2)*CKK+(r-q)*K*CK+q*C
    return res,{"C":C,"nu":nu,"sigma":sigma,"Cx":Cx,"CT":CT,"Cxx":Cxx,"K":K,"CK":CK,"CKK":CKK,"xT":xT}
