import argparse, torch, pandas as pd
from ..networks.wrappers import BaselineCall
from ..utils.checkpoint import load,peek_arch
from ..config import S0, r, q

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",required=True)
    p.add_argument("--in",dest="inp",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--width",type=int,default=0)
    p.add_argument("--hidden",type=int,default=0)
    p.add_argument("--bs0",type=float,default=0.25)
    args=p.parse_args()
    if args.width<=0 or args.hidden<=0:
        args.width,args.hidden=peek_arch(args.ckpt)
    model=BaselineCall(width=args.width,hidden=args.hidden,bs0=args.bs0,S0=S0,r=r,q=q).to(dtype=torch.float64)
    load(args.ckpt,model)
    df=pd.read_csv(args.inp)
    K=torch.tensor(df["K"].values,dtype=torch.float64)
    T=torch.tensor(df["T"].values,dtype=torch.float64)
    x=torch.log(K/S0)
    C,nu,sigma=model(torch.stack([x,T],-1))
    pd.DataFrame({"K":df["K"],"T":df["T"],"C":C.detach().numpy(),"nu":nu.detach().numpy(),"sigma":sigma.detach().numpy()}).to_csv(args.out,index=False)
    print(args.out)

if __name__=="__main__":
    main()
