import torch
class Boundaries:
    def __init__(self,x0T,xinfT,xT0):
        self.x0T=torch.as_tensor(x0T,dtype=torch.float64)
        self.xinfT=torch.as_tensor(xinfT,dtype=torch.float64)
        self.xT0=torch.as_tensor(xT0,dtype=torch.float64)
