import torch

def huber(x,delta):
    ax=x.abs()
    m=ax<=delta
    return torch.where(m,0.5*x.pow(2),delta*(ax-0.5*delta))
