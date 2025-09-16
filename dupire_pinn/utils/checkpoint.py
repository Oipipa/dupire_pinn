import torch

def save(path,model,opt=None,extra=None):
    d={"model":model.state_dict()}
    if opt is not None: d["opt"]=opt.state_dict()
    if extra is not None: d["extra"]=extra
    torch.save(d,path)

def load(path,model,opt=None):
    d=torch.load(path,map_location="cpu")
    model.load_state_dict(d["model"])
    if opt is not None and "opt" in d: opt.load_state_dict(d["opt"])
    return d.get("extra",None)

def peek_arch(path):
    d=torch.load(path,map_location="cpu")
    sd=d["model"]
    pref=None
    for p in ("trunk.0.weight","inner.trunk.0.weight"):
        if p in sd: pref="" if p.startswith("trunk.") else "inner."
    if pref is None:
        for k in sd.keys():
            if k.endswith("trunk.0.weight"):
                pref=k[:-len("trunk.0.weight")]
                break
    key=f"{pref}trunk.0.weight"
    width=sd[key].shape[0]
    trunk_ws=[k for k in sd.keys() if k.startswith(f"{pref}trunk.") and k.endswith(".weight")]
    hidden=len(trunk_ws)-1
    return int(width),int(hidden)
