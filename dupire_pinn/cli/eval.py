import argparse,torch,pandas as pd
from ..eval.grids import grid
from ..eval.metrics import violations,norm_residual,mass_error
from ..networks.wrappers import BaselineCall
from ..utils.checkpoint import load,peek_arch
from ..config import S0,r,q

def infer_domain_from_csv(csv,S0,margin=0.05):
    df=pd.read_csv(csv)
    Kmin=float(df["K"].quantile(0.02)); Kmax=float(df["K"].quantile(0.98))
    Tmin=float(df["T"].quantile(0.02)); Tmax=float(df["T"].quantile(0.98))
    xmin=float(torch.log(torch.tensor(Kmin/S0)).item())
    xmax=float(torch.log(torch.tensor(Kmax/S0)).item())
    dx=(xmax-xmin)*margin
    return xmin-dx,xmax+dx,Tmin,Tmax

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=str,required=True)
    p.add_argument("--nx",type=int,default=128)
    p.add_argument("--nt",type=int,default=64)
    p.add_argument("--eps",type=float,default=1e-6)
    p.add_argument("--ckpt",type=str,default="ckpt.pt")
    p.add_argument("--width",type=int,default=0)
    p.add_argument("--hidden",type=int,default=0)
    p.add_argument("--bs0",type=float,default=0.25)
    args=p.parse_args()
    xmin,xmax,Tmin,Tmax=infer_domain_from_csv(args.csv,S0)
    w,h=(args.width,args.hidden)
    if w<=0 or h<=0: w,h=peek_arch(args.ckpt)
    model=BaselineCall(width=w,hidden=h,bs0=args.bs0,S0=S0,r=r,q=q).to(dtype=torch.float64)
    load(args.ckpt,model)
    xg,Tg,Kg,_,_ = grid(xmin,xmax,Tmin,Tmax,args.nx,args.nt,S0)
    nv=violations(model,xg,Tg,args.eps,S0,r,q)
    nr=norm_residual(model,xg,Tg,S0,r,q).detach()
    me=mass_error(model,torch.linspace(xmin,xmax,args.nx,dtype=torch.float64),torch.linspace(Tmin,Tmax,args.nt,dtype=torch.float64),S0,r,q).detach()
    print(int(nv),float(nr),float(me))
if __name__=="__main__":
    main()
