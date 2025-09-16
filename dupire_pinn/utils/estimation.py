import numpy as np
import pandas as pd

def clean_cols(df):
    df.columns=[c.strip(" []") for c in df.columns]
    return df.rename(columns={"UNDERLYING_LAST":"underlying","EXPIRE_DATE":"expiry","QUOTE_DATE":"date","DTE":"dte","C_BID":"c_bid","C_ASK":"c_ask","P_BID":"p_bid","P_ASK":"p_ask","STRIKE":"strike"})

def mid(x,y):
    z=(x+y)/2.0
    return z.replace([np.inf,-np.inf],np.nan)

def estimate_S0(df):
    return float(np.nanmedian(df["underlying"].astype(float)))

def estimate_flat_rq(df,S0,tmin=0.01,tmax=2.0,min_bid=0.0,kq_lo=0.02,kq_hi=0.98,min_pts=8):
    d=df.copy()
    d["T"]=d["dte"]/365.0
    d=d[(d["T"]>=tmin)&(d["T"]<=tmax)]
    d=d.assign(C_mid=mid(d["c_bid"],d["c_ask"]),P_mid=mid(d["p_bid"],d["p_ask"]))
    if min_bid>0:
        d=d[(d["c_bid"]>=min_bid)|(d["c_ask"]>=min_bid)|(d["p_bid"]>=min_bid)|(d["p_ask"]>=min_bid)]
    out=[]
    for T,grp in d.groupby("T"):
        g=grp.copy()
        if g.shape[0]<min_pts: continue
        klo,gklo = g["strike"].quantile(kq_lo), g["strike"].quantile(kq_hi)
        g=g[(g["strike"]>=klo)&(g["strike"]<=gklo)]
        if g.shape[0]<min_pts: continue
        K=g["strike"].to_numpy(dtype=float)
        y=(g["C_mid"]-g["P_mid"]).to_numpy(dtype=float)
        if np.any(~np.isfinite(y)): continue
        A=np.vstack([np.ones_like(K),K]).T
        sol,_,_,_=np.linalg.lstsq(A,y,rcond=None)
        a,b=sol[0],sol[1]
        D=-b
        if not np.isfinite(D) or D<=0 or D>1.2: continue
        F=a/D
        if not np.isfinite(F) or F<=0: continue
        r_T=-np.log(max(D,1e-12))/max(T,1e-12)
        mu_T=np.log(max(F,1e-12)/S0)/max(T,1e-12)
        q_T=r_T-mu_T
        out.append((T,r_T,q_T,F,D))
    if not out:
        return None,None
    arr=np.array(out)
    r_hat=float(np.nanmedian(arr[:,1]))
    q_hat=float(np.nanmedian(arr[:,2]))
    return r_hat,q_hat
