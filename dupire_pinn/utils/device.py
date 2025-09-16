import torch

def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
