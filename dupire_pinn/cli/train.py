import argparse, torch, pandas as pd
from ..networks.wrappers import BaselineCall
from ..config import S0, r, q, INT_N, SEED
from ..sampling import latin_hypercube
from ..data.collocation import Collocation
from ..data.boundaries import Boundaries
from ..data.market import Market
from ..utils.io import load_market_csv
from ..utils.seed import set_seed
from ..utils.checkpoint import save, load
from ..train.loop import train

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
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--nx",type=int,default=256)
    p.add_argument("--nt",type=int,default=256)
    p.add_argument("--al",action="store_true")
    p.add_argument("--eta",type=float,default=0.5)
    p.add_argument("--ckpt",type=str,default="ckpt.pt")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--sigma_ref",type=float,default=0.25)
    p.add_argument("--lam_pos",type=float,default=5e-3)
    p.add_argument("--lam_sigma",type=float,default=1e-3)
    p.add_argument("--lam_tie",type=float,default=1e-3)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--hidden",type=int,default=4)
    p.add_argument("--batch_int",type=int,default=4096)
    p.add_argument("--batch_mkt",type=int,default=2048)
    p.add_argument("--bs0",type=float,default=0.25)
    p.add_argument("--w_pde",type=float,default=0.02)
    p.add_argument("--w_mkt",type=float,default=1.0)
    p.add_argument("--w_arb",type=float,default=0.02)
    p.add_argument("--w_bc",type=float,default=0.10)
    p.add_argument("--w_reg1",type=float,default=1e-5)
    p.add_argument("--w_reg2",type=float,default=0.0)
    p.add_argument("--w_mass",type=float,default=0.0)
    p.add_argument("--mass_nx",type=int,default=256)
    p.add_argument("--mass_nt",type=int,default=12)
    args=p.parse_args()
    set_seed(SEED)
    xmin,xmax,Tmin,Tmax=infer_domain_from_csv(args.csv,S0)
    xT_lh=latin_hypercube(INT_N,2,torch.tensor([xmin,Tmin],dtype=torch.float64),torch.tensor([xmax,Tmax],dtype=torch.float64))
    x,T,C=load_market_csv(args.csv,S0); market=Market(x,T,C)
    sz=min(INT_N,market.x.numel())
    idx=torch.randint(0,market.x.numel(),(sz,))
    xm=market.x[idx]+0.01*torch.randn_like(market.x[idx])
    Tm=market.T[idx]+0.01*torch.rand_like(market.T[idx])
    xT_m=torch.stack([xm.clamp(xmin,xmax),Tm.clamp(Tmin,Tmax)],-1)
    xT=torch.cat([xT_lh,xT_m],0)
    ints=Collocation(xT[:,0],xT[:,1])
    x0T=torch.stack([torch.full((args.nt,),xmin,dtype=torch.float64),torch.linspace(Tmin,Tmax,args.nt,dtype=torch.float64)],-1)
    xinfT=torch.stack([torch.full((args.nt,),xmax,dtype=torch.float64),torch.linspace(Tmin,Tmax,args.nt,dtype=torch.float64)],-1)
    xT0=torch.stack([torch.linspace(xmin,xmax,args.nx,dtype=torch.float64),torch.full((args.nx,),Tmin,dtype=torch.float64)],-1)
    bd=Boundaries(x0T,xinfT,xT0)
    model=BaselineCall(width=args.width,hidden=args.hidden,bs0=args.bs0,S0=S0,r=r,q=q).to(dtype=torch.float64)
    if args.resume: load(args.ckpt,model)
    weights={"pde":args.w_pde,"mkt":args.w_mkt,"arb":args.w_arb,"bc":args.w_bc,"reg1":args.w_reg1,"reg2":args.w_reg2}
    xline_mass=torch.linspace(xmin,xmax,args.mass_nx,dtype=torch.float64)
    T_mass=torch.linspace(Tmin,Tmax,args.mass_nt,dtype=torch.float64)
    model=train(model,S0,r,q,ints,bd,market,weights,epochs=args.epochs,lr=args.lr,al=args.al,eta=args.eta,
                sigma_ref=args.sigma_ref,lam_pos=args.lam_pos,lam_sigma=args.lam_sigma,lam_tie=args.lam_tie,
                batch_int=args.batch_int,batch_mkt=args.batch_mkt,
                w_mass=args.w_mass,xline_mass=xline_mass,T_mass=T_mass)
    save(args.ckpt,model)

if __name__=="__main__":
    main()
