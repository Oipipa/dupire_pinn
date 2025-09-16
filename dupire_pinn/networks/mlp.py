import torch.nn as nn

def mlp(in_dim,hidden,width,out_dim,act='gelu'):
    a=nn.GELU() if act=='gelu' else nn.SiLU()
    layers=[nn.Linear(in_dim,width),a]
    for _ in range(hidden-1):
        layers+=[nn.Linear(width,width),a]
    layers.append(nn.Linear(width,out_dim))
    return nn.Sequential(*layers)
