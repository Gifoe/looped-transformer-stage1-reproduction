"""Frozen tests on complete generated integers, with target-clustered intervals."""
from collections import Counter

import numpy as np

SOURCES = ("joint","quarter","symmetric","source_geometry")
BACKGROUNDS = ("clean","symmetric")
ARMS = ("clean",*SOURCES,
        *(f"{bg}_restore_{part}{control}" for bg,parts in (("joint",("p",)),("symmetric",("p","m","both")))
          for part in parts for control in ("","_geometry")),
        *(f"{bg}_remove_{p}" for bg in BACKGROUNDS for p in ("p","m","both")),
        *(f"{bg}_dose_{p}{g}" for bg in BACKGROUNDS for p in ("p","m") for g in ("","_geometry")))


def paired(rows, values):
    bins = {}
    for x,v in zip(rows,values,strict=True):
        counts = bins.setdefault(x["target"],[0,0])
        counts[0] += v
        counts[1] += 1
    if not bins:
        return dict(n=0,targets=0,total=0,rate=None,interval=[None,None])
    a = np.array([bins[k] for k in sorted(bins)])
    ix = np.random.default_rng(261123011).integers(len(a),size=(20000,len(a)))
    sums = a[ix].sum(axis=1)
    d = sums[:,0]/sums[:,1]
    return dict(n=len(rows),targets=len(a),total=int(a[:,0].sum()),rate=float(a[:,0].sum()/len(rows)),
                interval=np.quantile(d,[.0125,.9875]).tolist())


def positive(a):
    return a["interval"][0] is not None and a["interval"][0]>0


def summarize(rows):
    groups = {}
    for part in ("repair","safe"):
        sub = [x for x in rows if x["kind"]==part]
        cases = {}
        for arm in ARMS:
            d = Counter(correct=0,foil=0,other=0,new_error=0,new_nonfoil=0,
                        source_losses=0,source_restored=0,new_restoration_harm=0)
            for r in sub:
                target,foil = r["target"],r["foil"]
                pred = {k:v["number"] for k,v in r["pred"].items()}
                x = pred[arm]
                d["correct"] += x==target
                d["foil"] += x==foil
                d["other"] += x not in (target,foil)
                d["new_error"] += pred["clean"]==target and x!=target
                d["new_nonfoil"] += pred["clean"]==target and x not in (target,foil)
                if "_restore_" in arm:
                    parent = pred[arm.split("_restore_",1)[0]]
                    d["source_losses"] += pred["clean"]==target and parent==foil
                    d["source_restored"] += pred["clean"]==target and parent==foil and x==target
                    d["new_restoration_harm"] += parent==target and x!=target
            cases[arm] = dict(d)
        aligned = [x for x in sub if x["aligned"]]
        response = {}
        for name in SOURCES:
            if aligned:
                response[name] = {k:float(np.median([x["shifts"][name][k] for x in aligned]))
                                  for k in aligned[0]["shifts"][name]}
        groups[part] = dict(n=len(sub),aligned=len(aligned),matched=sum(all(x is not None and x<=2 for x in r["match"]) for r in sub),
                            counts=cases,median_response=response)
    rep = [x for x in rows if x["kind"]=="repair"]
    def exact(x,a,which):
        return x["pred"][a]["number"]==x[which]

    def contrast(a,b,endpoint):
        return paired(rep,[int(exact(x,a,endpoint))-int(exact(x,b,endpoint)) for x in rep])

    def rescued(x,a):
        bg = a.split("_restore_",1)[0]
        return exact(x,"clean","target") and exact(x,bg,"foil") and exact(x,a,"target")

    def restore_contrast(a,b):
        return paired(rep,[int(rescued(x,a))-int(rescued(x,b)) for x in rep])

    for bg in BACKGROUNDS:
        prefix = bg+"_dose_"
        per = dict(p_vs_m=contrast(prefix+"p",prefix+"m","foil"),
                   m_vs_p=contrast(prefix+"m",prefix+"p","foil"),
                   p_vs_geometry=contrast(prefix+"p",prefix+"p_geometry","foil"),
                   m_vs_geometry=contrast(prefix+"m",prefix+"m_geometry","foil"),
                   discrete_interaction=paired(rep,[int(exact(x,bg+"_remove_both","foil"))
                       -int(exact(x,bg+"_remove_p","foil"))-int(exact(x,bg+"_remove_m","foil"))
                       +int(exact(x,bg,"foil")) for x in rep]))
        per["dominance"] = {a:groups["repair"]["counts"][prefix+a]["foil"]>=8 and positive(per[f"{a}_vs_{b}"])
                            and positive(per[f"{a}_vs_geometry"]) for a,b in (("p","m"),("m","p"))}
        groups[bg+"_factorial"] = per
    dp = contrast("symmetric","source_geometry","foil")
    out = dict(symmetric_loss_vs_geometry=dp,
               joint_loss_vs_quarter=contrast("joint","quarter","foil"),
               symmetric_loss_vs_quarter=contrast("symmetric","quarter","foil"),
               joint_p_vs_geometry=restore_contrast("joint_restore_p","joint_restore_p_geometry"))
    safe = [x for x in rows if x["kind"]=="safe"]
    assert [x["id"] for x in rep]==[x["id"] for x in safe]
    out["symmetric_same_vector_borrow_minus_safe"] = paired(rep,[int(exact(x,"symmetric","foil"))
         -int(exact(x,"source_geometry","foil"))-int(exact(y,"symmetric","foil"))
         +int(exact(y,"source_geometry","foil")) for x,y in zip(rep,safe,strict=True)])
    gates = {}
    for a,b,key in (("p","m","P_shift"),("m","p","M_shift")):
        restored = "symmetric_restore_"+a
        control = restore_contrast(restored,restored+"_geometry")
        other = restore_contrast(restored,"symmetric_restore_"+b)
        moved = sum(x["aligned"] and x["shifts"]["symmetric"][key]>1e-3 and
                    exact(x,"symmetric","foil") for x in rep)
        out[f"symmetric_restore_{a}_vs_geometry"] = control
        out[f"symmetric_restore_{a}_vs_{b}"] = other
        out[f"symmetric_reverted_with_{a}_shift"] = moved
        gates[a] = groups["repair"]["counts"]["symmetric"]["foil"]>=8 and moved>=8 and all(map(positive,(dp,control,other)))
    groups["source_contrasts"] = out
    groups["gates"] = gates
    return groups
