import torch

def no_arb_hinge(derivs):
    CK=derivs["CK"]
    CKK=derivs["CKK"]
    CT=derivs["CT"]
    return torch.relu(CK).pow(2).mean()+torch.relu(-CKK).pow(2).mean()+torch.relu(-CT).pow(2).mean()
