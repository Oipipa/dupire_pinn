import torch,random,os,numpy as np

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"]=str(s)
