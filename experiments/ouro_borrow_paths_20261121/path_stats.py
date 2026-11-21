"""Prespecified whole-answer estimates and a cluster-level exclusion bound."""
from collections import defaultdict
from statistics import median

from edge_stats import upper_binomial
from origin_stats import paired, positive

PATHS=("context","pivot","joint")
ARMS=("native",*PATHS,*(p+s for p in PATHS for s in ("_capped","_matched")),
      *("restore_"+p for p in PATHS))


def verdict(r,k,endpoint):
    return r["pred"][k]["number"]==r[endpoint]


def damage(r,k):
    return verdict(r,"native","target") and not verdict(r,k,"target")


def stats(rows):
    counts={}
    for k in ARMS:
        counts[k]=dict(correct=sum(verdict(r,k,"target") for r in rows),
                       foil=sum(verdict(r,k,"foil") for r in rows),
                       other_new=sum(damage(r,k) and not verdict(r,k,"foil") for r in rows),
                       any_new=sum(damage(r,k) for r in rows),
                       restored=sum(verdict(r,"native","target") and verdict(r,"joint","foil")
                                    and verdict(r,k,"target") for r in rows))
    use=[r for r in rows if r["aligned"]]
    receipts={}
    for path in PATHS:
        frac=lambda v:v["matched"]/max(v["full"],1e-12)
        if use:
            receipts[path]=dict(
                dose_fraction=median(frac(r["masses"][path]) for r in use),
                relative_available=median(r["masses"][path]["matched"]/max(r["masses"][path]["available"],1e-12)
                                          for r in use),
                query_count=median(r["query_counts"][path] for r in use))
    margins={k:median(r["margin"][k]-r["margin"]["native"] for r in use) for k in ARMS} if use else {}
    return dict(n=len(rows),aligned=len(use),counts=counts,mass=receipts,margin_shift=margins)


def core(rep,safe):
    assert [r["id"] for r in rep]==[r["id"] for r in safe]
    z={"repair":stats(rep),"safe":stats(safe)}
    compare={}
    gates={}
    for p in PATHS:
        matched=paired(rep,[int(verdict(r,p+"_capped","foil"))-int(verdict(r,p+"_matched","foil"))
                            for r in rep])
        safe_effect=paired(rep,[int(damage(a,p))-int(damage(b,p)) for a,b in zip(rep,safe,strict=True)])
        compare[p]=dict(mass_equalized=matched,repair_minus_safe=safe_effect)
        gates[p]=z["repair"]["counts"][p]["foil"]>=8 and positive(matched) and positive(safe_effect)
    foil=lambda r,k:int(verdict(r,k,"foil"))
    compare["factorial_reversion"]=paired(rep,[foil(r,"joint")-foil(r,"context")
                                                -foil(r,"pivot")+foil(r,"native") for r in rep])
    compare["pivot_minus_joint"]=paired(rep,[foil(r,"pivot")-foil(r,"joint") for r in rep])
    compare["pivot_minus_context"]=paired(rep,[foil(r,"pivot")-foil(r,"context") for r in rep])
    compare["restore_context_minus_pivot"]=paired(rep,[
        (int(verdict(r,"restore_context","target"))-int(verdict(r,"restore_pivot","target")))
        if foil(r,"joint") else 0 for r in rep])
    eq=compare["pivot_minus_joint"]["interval"]
    gates["pivot_dominant"]=gates["pivot"] and eq[0]>=-.05 and eq[1]<=.05 and positive(compare["pivot_minus_context"])
    clusters=defaultdict(bool)
    for r in rep:
        clusters[r["target"]]|=bool(foil(r,"joint"))
    z["upper"]=dict(targets=len(clusters),events=sum(clusters.values()),
                   upper_97_5=upper_binomial(sum(clusters.values()),len(clusters)))
    gates["exclude_joint_10pct"]=bool(rep) and z["repair"]["aligned"]==len(rep) and z["upper"]["upper_97_5"]<.1
    z["paired"]=compare
    z["gates"]=gates
    return z


def summarize(rows):
    rep,safe=([r for r in rows if r["kind"]==k] for k in ("repair","safe"))
    result=core(rep,safe)
    distances={r["id"] for r in rep if all(x is not None and x<=2. for x in r["distance"])}
    result["margin_matched"]=core([r for r in rep if r["id"] in distances],[r for r in safe if r["id"] in distances])
    taken=[r["components"] for r in rep if "components" in r]
    result["components"]=dict(n=len(taken),medians={a:{k:median(r[a][k] for r in taken) for k in p}
                               for a,p in taken[0].items()} if taken else {})
    d={}
    for k in ("3","4"):
        depth=[(r,r["depth"][k]) for r in rep if "depth" in r]
        t={}
        for a in ("native",*PATHS):
            t[a]=dict(correct=sum(p["pred"][a]["number"]==r["target"] for r,p in depth),
                      foil=sum(p["pred"][a]["number"]==r["foil"] for r,p in depth),
                      harm=sum(p["pred"]["native"]["number"]==r["target"] and p["pred"][a]["number"]!=r["target"]
                               for r,p in depth))
        d[k]=dict(n=len(depth),aligned=sum(p["aligned"] for _,p in depth),arms=t)
    result["deeper"]=d
    return result
