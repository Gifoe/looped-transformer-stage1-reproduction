"""AB1 boundary/region arithmetic; preparation only, not a GPU study."""
import hashlib
import torch

if not __debug__: raise RuntimeError("optimized Python forbidden")

def _validate(E,B,prefix,R):
    if E.ndim!=3 or E.shape[0]!=1 or E.shape!=B.shape or E.dtype!=B.dtype or E.device!=B.device: raise ValueError("incompatible E/B")
    if not (0<prefix<R==E.shape[1]-1): raise ValueError("invalid causal regions")
    if not torch.isfinite(E).all() or not torch.isfinite(B).all(): raise ValueError("nonfinite source")

def corner(E,B,prefix,R,q_source,r_source):
    _validate(E,B,prefix,R)
    if q_source not in ("E","B") or r_source not in ("E","B"): raise ValueError("source")
    out=B.clone();out[:,prefix:R,:]=(E if q_source=="E" else B)[:,prefix:R,:];out[:,R,:]=(E if r_source=="E" else B)[:,R,:]
    if not torch.equal(out[:,:prefix,:],B[:,:prefix,:]): raise AssertionError("prefix modified")
    return out

def q9_cross(receiver,donor,prefix,R):
    _validate(receiver,donor,prefix,R);out=receiver.clone();out[:,prefix:R,:]=donor[:,prefix:R,:]
    if not torch.equal(out[:,:prefix,:],receiver[:,:prefix,:]) or not torch.equal(out[:,R,:],receiver[:,R,:]): raise AssertionError("cross outside Q")
    return out

def control(receiver,donor,prefix,R,region,kind,split,ident,edge):
    _validate(receiver,donor,prefix,R)
    if region not in ("Q","R") or kind not in ("norm","random"): raise ValueError("control")
    positions=range(prefix,R) if region=="Q" else (R,);out=receiver.clone();rows=[]
    for t in positions:
        a=receiver[0,t].detach().cpu().double();b=donor[0,t].detach().cpu().double();delta=b-a;dose=delta.norm();na=a.norm();nb=b.norm()
        if kind=="norm":
            if nb==0 and na!=0: return None,{"status":"UNDEFINED_ZERO_DONOR", "position":t,"partial_not_applied":True}
            exact=b if nb==0 else b*(na/nb); seed=None
        else:
            seed=int.from_bytes(hashlib.sha256((f"AB1-v1|{split}|{ident}|{edge}|{t}").encode()).digest()[:8],"big") & ((1<<63)-1)
            if dose==0: exact=a.clone()
            else:
                gen=torch.Generator(device="cpu");gen.manual_seed(seed);direction=torch.randn(a.shape,generator=gen,dtype=torch.float64);n=direction.norm()
                if n==0: return None,{"status":"UNDEFINED_ZERO_RANDOM", "position":t,"partial_not_applied":True}
                exact=a+direction*(dose/n)
        value=exact.to(receiver.dtype);out[0,t]=value.to(receiver.device)
        rows.append({"position":t,"seed":seed,"receiver_norm":float(na),"donor_norm":float(nb),"nominal_displacement":float(dose),"actual_displacement":float((value.double()-a).norm()),"actual_norm":float(value.double().norm()),"cast_residual":(value.double()-exact)})
    outside=torch.ones(receiver.shape[1],dtype=torch.bool,device=receiver.device);outside[list(positions)]=False
    if not torch.equal(out[:,outside],receiver[:,outside]): raise AssertionError("nonintervened changed")
    return out,{"status":"OK","rows":rows}
