import torch

def _norm_cdf(x):
    return 0.5*(1.0+torch.erf(x/torch.sqrt(torch.tensor(2.0,dtype=x.dtype,device=x.device))))

def bs_call(S0,K,T,r,q,sigma):
    eps=1e-12
    T=torch.clamp(T,min=eps)
    sig=torch.clamp(sigma,min=eps)
    d1=(torch.log(S0/K)+(r-q+0.5*sig*sig)*T)/(sig*torch.sqrt(T))
    d2=d1-sig*torch.sqrt(T)
    return S0*torch.exp(-q*T)*_norm_cdf(d1)-K*torch.exp(-r*T)*_norm_cdf(d2)
