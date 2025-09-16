import torch

def latin_hypercube(n,dim,low,high,seed=1234):
    g=torch.Generator().manual_seed(seed)
    cut=torch.linspace(0,1,n+1)
    u=torch.empty(n,dim).uniform_(generator=g)
    a=cut[:-1].unsqueeze(1)
    b=cut[1:].unsqueeze(1)
    rd=a+(b-a)*u
    for j in range(dim):
        idx=torch.randperm(n,generator=g)
        rd[:,j]=rd[idx,j]
    return low+(high-low)*rd
