import math

def ramp(ep,start,end,ep_start,ep_end):
    if ep<=ep_start: return start
    if ep>=ep_end: return end
    t=(ep-ep_start)/(ep_end-ep_start)
    return start+(end-start)*t

def cosine_factor(ep,total):
    if total<=0: return 1.0
    return 0.5*(1+math.cos(math.pi*ep/total))
