import torch.optim as optim

def make_optim(params,lr=1e-3,weight_decay=0.0):
    return optim.AdamW(params,lr=lr,weight_decay=weight_decay)
