import torch

class Market:
    def __init__(self,x,T,C):
        self.x=torch.as_tensor(x,dtype=torch.float64)
        self.T=torch.as_tensor(T,dtype=torch.float64)
        self.C=torch.as_tensor(C,dtype=torch.float64)
