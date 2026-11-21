"""Exact whole-integer reasoning outcomes and target-clustered tests."""
from collections import defaultdict
from statistics import median

from edge_stats import upper_binomial
from origin_stats import paired, positive

CONTEXTS=("receiver","opposite","same")
MODES={b+"_"+m:(b,m) for b in CONTEXTS for m in ("native","donor","shuffle")}
ARMS=tuple(MODES)
NUMERIC=("target","foil","mixed","plus","minus")


def hit(r,a,k):
    return r["pred"][a]["number"]==r[k]


def outcomes(rows):
    arms={}
    for a in ARMS:
        ans={k:sum(hit(r,a,k) for r in rows) for k in NUMERIC}
        ans["new_error"]=sum(hit(r,"receiver_native","target") and not hit(r,a,"target") for r in rows)
        ans["unparsed"]=sum(r["pred"][a]["number"] is None for r in rows)
        ans["other"]=sum(r["pred"][a]["number"] is not None and not any(hit(r,a,k) for k in NUMERIC) for r in rows)
        arms[a]=ans
    native={a:[sum(r["donor_native"][a][j]["number"]==r[k] for r in rows) for j in (0,1)]
            for a,k in (("opposite","plus"),("same","minus"))}
    return dict(n=len(rows),arms=arms,donor_native=native)


def core(repair,safe):
    assert [r["id"] for r in repair]==[r["id"] for r in safe]
    out={k:outcomes(r) for k,r in (("repair",repair),("safe",safe))}
    contrasts={}
    bounds={}
    gates={}
    def excess(r,a):
        return int(hit(r,a+"_donor","mixed"))-int(hit(r,a+"_shuffle","mixed"))

    for a in CONTEXTS:
        contrasts[a+"_mixed_vs_shuffle"]=paired(repair,[excess(x,a) for x in repair])
        contrasts[a+"_safe_mixed_vs_shuffle"]=paired(safe,[excess(x,a) for x in safe])
        contrasts[a+"_mixed_vs_native"]=paired(repair,[int(hit(r,a+"_donor","mixed"))-int(hit(r,a+"_native","mixed"))
                                                       for r in repair])
        cluster=defaultdict(bool)
        for x in repair:
            cluster[x["target"]]|=excess(x,a)>0
        e,n=sum(cluster.values()),len(cluster)
        bounds[a]=dict(n=n,events=e,upper_97_5=upper_binomial(e,n))
    for a,b in (("opposite","receiver"),("same","receiver"),("opposite","same")):
        contrasts[a+"_minus_"+b]=paired(repair,[excess(x,a)-excess(x,b) for x in repair])
    gates["composition"]=all(out["repair"]["arms"][a+"_donor"]["mixed"]>=8 and
                             positive(contrasts[a+"_mixed_vs_shuffle"]) for a in ("receiver","opposite"))
    lo,hi=contrasts["opposite_minus_receiver"]["interval"]
    gates["context_competition"]=bool(repair) and gates["composition"] and (lo>0 or hi<0)
    gates["exclude_large_mixture"]=bool(repair) and all(q["upper_97_5"]<.1 for q in bounds.values())
    out.update(paired=contrasts,bounds=bounds,gates=gates)
    return out


def summarize(rows):
    repair,safe=([r for r in rows if r["kind"]==k] for k in ("repair","safe"))
    out=core(repair,safe)
    good={r["id"] for r in repair if all(v is not None and v<=2 for v in r["distance"])}
    out["margin_matched"]=core([r for r in repair if r["id"] in good],[r for r in safe if r["id"] in good])
    values=[r["diagnostics"] for r in repair if "diagnostics" in r]
    out["diagnostic"]=dict(n=len(values),median={a:{k:median(r[a][k] for r in values) for k in q}
                                               for a,q in values[0].items()} if values else {})
    return out
