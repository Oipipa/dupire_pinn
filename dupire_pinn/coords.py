import torch

def x_from_K(K,S0):
    return torch.log(K/S0)

def K_from_x(x,S0):
    return S0*torch.exp(x)

def CK_from_Cx(Cx,K):
    return Cx/K

def CKK_from_Cxx_Cx(Cxx,Cx,K):
    return (Cxx-Cx)/K.pow(2)
