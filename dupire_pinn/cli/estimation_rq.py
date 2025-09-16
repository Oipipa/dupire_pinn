import argparse, pandas as pd
from ..utils.estimation import clean_cols, estimate_S0, estimate_flat_rq

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--in",dest="inp",required=True)
    p.add_argument("--date",type=str,required=True)
    p.add_argument("--tmin",type=float,default=0.01)
    p.add_argument("--tmax",type=float,default=2.0)
    p.add_argument("--min_bid",type=float,default=0.0)
    p.add_argument("--kq_lo",type=float,default=0.02)
    p.add_argument("--kq_hi",type=float,default=0.98)
    args=p.parse_args()
    df=clean_cols(pd.read_csv(args.inp))
    df=df[df["date"].astype(str).str.strip()==args.date]
    S0=estimate_S0(df)
    r,q=estimate_flat_rq(df,S0,args.tmin,args.tmax,args.min_bid,args.kq_lo,args.kq_hi)
    print(f"S0={S0}")
    print(f"r={r}")
    print(f"q={q}")

if __name__=="__main__":
    main()
