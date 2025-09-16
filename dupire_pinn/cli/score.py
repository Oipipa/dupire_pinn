import argparse
import torch
from ..networks.wrappers import BaselineCall
from ..utils.checkpoint import load,peek_arch
from ..utils.io import load_market_csv
from ..data.market import Market
from ..eval.metrics import rmse_market
from ..config import S0,r,q

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",type=str,required=True)
    p.add_argument("--csv",type=str,required=True)
    p.add_argument("--width",type=int,default=0)
    p.add_argument("--hidden",type=int,default=0)
    p.add_argument("--bs0",type=float,default=0.25)
    args=p.parse_args()
    w,h=(args.width,args.hidden)
    if w<=0 or h<=0:
        w,h=peek_arch(args.ckpt)
    model=BaselineCall(width=w,hidden=h,bs0=args.bs0,S0=S0,r=r,q=q).to(dtype=torch.float64)
    load(args.ckpt,model)
    x,T,C=load_market_csv(args.csv,S0)
    mkt=Market(x,T,C)
    rmse=rmse_market(model,mkt).item()
    print(f"RMSE_market_abs={rmse:.6f}")
    print(f"RMSE_market_rel={rmse/max(1e-9,S0):.6f}")

if __name__=="__main__":
    main()
