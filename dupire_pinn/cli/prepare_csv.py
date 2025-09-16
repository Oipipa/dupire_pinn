import argparse
import pandas as pd
import numpy as np

def clean_cols(df):
    df.columns=[c.strip(" []") for c in df.columns]
    return df.rename(columns={"UNDERLYING_LAST":"underlying","EXPIRE_DATE":"expiry","QUOTE_DATE":"date","DTE":"dte","C_BID":"c_bid","C_ASK":"c_ask","P_BID":"p_bid","P_ASK":"p_ask","STRIKE":"strike"})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--in",dest="inp",type=str,required=True)
    p.add_argument("--date",type=str,default=None)
    p.add_argument("--out",type=str,required=True)
    p.add_argument("--tmin",type=float,default=0.01)
    p.add_argument("--tmax",type=float,default=2.0)
    p.add_argument("--min_bid",type=float,default=0.0)
    p.add_argument("--kq_lo",type=float,default=0.02)
    p.add_argument("--kq_hi",type=float,default=0.98)
    args=p.parse_args()
    df=clean_cols(pd.read_csv(args.inp))
    df["date"]=df["date"].astype(str).str.strip()
    use_date=args.date or df["date"].iloc[0]
    d=df[df["date"]==use_date].copy()
    d["mid_c"]=(d["c_bid"]+d["c_ask"])/2.0
    d["mid_p"]=(d["p_bid"]+d["p_ask"])/2.0
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=["mid_c","strike","underlying","dte"])
    d=d[d["dte"]>0]
    if args.min_bid>0: d=d[(d["c_bid"]>=args.min_bid)|(d["c_ask"]>=args.min_bid)]
    S0=float(d["underlying"].median())
    d["T"]=d["dte"]/365.0
    d=d[(d["T"]>=args.tmin)&(d["T"]<=args.tmax)]
    klo, khi = d["strike"].quantile(args.kq_lo), d["strike"].quantile(args.kq_hi)
    d=d[(d["strike"]>=klo)&(d["strike"]<=khi)]
    out=d[["strike","T","mid_c"]].rename(columns={"strike":"K","mid_c":"C"}).sort_values(["T","K"]).reset_index(drop=True)
    out.to_csv(args.out,index=False)
    print(args.out)
    print(f"DATE={use_date}")
    print(f"S0={S0}")
if __name__=="__main__":
    main()
