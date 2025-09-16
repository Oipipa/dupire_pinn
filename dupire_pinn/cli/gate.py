import argparse, torch
from ..networks.wrappers import BaselineCall
from ..utils.checkpoint import load,peek_arch
from ..utils.io import load_market_csv
from ..data.market import Market
from ..eval.metrics import rmse_market, violations, norm_residual, mass_error
from ..eval.grids import grid
from ..config import S0, r, q

def infer_domain_from_csv(csv,S0,margin=0.05):
    import pandas as pd
    df=pd.read_csv(csv)
    Kmin=float(df["K"].quantile(0.02)); Kmax=float(df["K"].quantile(0.98))
    Tmin=float(df["T"].quantile(0.02)); Tmax=float(df["T"].quantile(0.98))
    xmin=float(torch.log(torch.tensor(Kmin/S0)).item())
    xmax=float(torch.log(torch.tensor(Kmax/S0)).item())
    dx=(xmax-xmin)*margin
    return xmin-dx,xmax+dx,Tmin,Tmax

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",required=True)
    p.add_argument("--csv",required=True)
    p.add_argument("--nx",type=int,default=128)
    p.add_argument("--nt",type=int,default=64)
    p.add_argument("--max_nv",type=int,default=0)
    p.add_argument("--max_resid",type=float,default=1e-3)
    p.add_argument("--max_mass",type=float,default=1e-3)
    p.add_argument("--max_rmse_rel",type=float,default=1e-3)
    p.add_argument("--width",type=int,default=0)
    p.add_argument("--hidden",type=int,default=0)
    p.add_argument("--bs0",type=float,default=0.25)
    args=p.parse_args()
    if args.width<=0 or args.hidden<=0:
        args.width,args.hidden=peek_arch(args.ckpt)
    model=BaselineCall(width=args.width,hidden=args.hidden,bs0=args.bs0,S0=S0,r=r,q=q).to(dtype=torch.float64)
    load(args.ckpt,model)
    xmin,xmax,Tmin,Tmax=infer_domain_from_csv(args.csv,S0)
    xg,Tg,Kg,xline,Tline=grid(xmin,xmax,Tmin,Tmax,args.nx,args.nt,S0)
    nv=int(violations(model,xg,Tg,1e-6,S0,r,q))
    nr=float(norm_residual(model,xg,Tg,S0,r,q))
    me=float(mass_error(model,xline,Tline,S0,r,q))
    x,T,C=load_market_csv(args.csv,S0)
    mkt=Market(x,T,C)
    rmse_abs=float(rmse_market(model,mkt))
    rmse_rel=rmse_abs/max(1e-12,S0)
    print(f"nv={nv} resid={nr} mass={me} rmse_abs={rmse_abs} rmse_rel={rmse_rel}")
    ok=(nv<=args.max_nv) and (nr<=args.max_resid) and (me<=args.max_mass) and (rmse_rel<=args.max_rmse_rel)
    raise SystemExit(0 if ok else 1)

if __name__=="__main__":
    main()
