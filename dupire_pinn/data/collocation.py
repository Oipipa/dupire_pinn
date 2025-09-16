import torch
class Collocation:
    def __init__(self,x,T):
        self.x=torch.as_tensor(x,dtype=torch.float64)
        self.T=torch.as_tensor(T,dtype=torch.float64)
