import pandas as pd
import torch

def load_market_csv(path,S0,cols=("K","T","C")):
    df=pd.read_csv(path)
    K=torch.as_tensor(df[cols[0]].values,dtype=torch.float64)
    T=torch.as_tensor(df[cols[1]].values,dtype=torch.float64)
    C=torch.as_tensor(df[cols[2]].values,dtype=torch.float64)
    x=torch.log(K/S0)
    return x,T,C
