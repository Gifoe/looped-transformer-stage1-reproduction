"""Paired behavioral effects for residual contents and downstream recomputation."""
from collections import defaultdict

from edge_stats import upper_binomial
from origin_stats import paired, positive

SIDES=("original","swapped")
CONTROLS=("geometry","irrelevant")
CONFIG={a+"_"+w:(a,w) for w in ("adaptive","locked") for a in ("own","swap",*CONTROLS)}
CONFIG.update(own_fixed_other=("own","fixed_other"),swap_fixed_other=("swap","fixed_other"))
ARMS=("native",*CONFIG)


def exact(r,s,a,other=False):
    k="target" if (s=="original")^other else "old"
    return r["pred"][s][a]["number"]==r[k]


def summarize(rows):
    outcome={}
    for s in SIDES:
        outcome[s]={}
        for a in ARMS:
            t=sum(exact(r,s,a) for r in rows)
            d=sum(exact(r,s,a,True) for r in rows)
            un=sum(r["pred"][s][a]["number"] is None for r in rows)
            outcome[s][a]=dict(receiving=t,other_table=d,other_value=len(rows)-t-d-un,unparsed=un)
    du={a:sum(all(exact(r,s,a,True) for s in SIDES) for r in rows) for a in ARMS}
    supports={}
    stops={}
    masks=[(r,s) for r in rows for s in SIDES]
    unit=[r for r,_ in masks]
    def bound(flags):
        v=defaultdict(bool)
        for r,k in flags:
            v[r["target"]]|=k>0
        n,e=len(v),sum(v.values())
        return dict(n=n,events=e,upper_97_5=upper_binomial(e,n))

    def rescue(r,s,k):
        return int(exact(r,s,k+"_adaptive"))-int(exact(r,s,k+"_locked"))

    swapped=[rescue(r,s,"swap") for r,s in masks]
    for k in CONTROLS:
        better=[int(all(exact(r,s,"swap_locked",True) for s in SIDES))
                -int(all(exact(r,s,k+"_locked",True) for s in SIDES)) for r in rows]
        comp=[x-rescue(r,s,k) for x,(r,s) in zip(swapped,masks,strict=True)]
        supports[k]=dict(residual=paired(rows,better),compensation=paired(unit,comp))
        stops[k]=dict(residual=bound(zip(rows,better,strict=True)),compensation=bound(zip(unit,comp,strict=True)))
    n_help=sum(x>0 for x in swapped)
    n_harm=sum(x<0 for x in swapped)
    gates=dict(residual=du["swap_locked"]>=8 and all(positive(v["residual"]) for v in supports.values()),
               compensation=n_help-n_harm>=8 and all(positive(v["compensation"]) for v in supports.values()),
               exclude_large_residual=bool(rows) and all(v["residual"]["upper_97_5"]<.1 for v in stops.values()),
               exclude_large_compensation=bool(rows) and all(v["compensation"]["upper_97_5"]<.1 for v in stops.values()))
    conditional={}
    for a in ("own","swap"):
        x,y=a+"_fixed_other",a+"_locked"
        diff=[int(exact(r,s,x,True))-int(exact(r,s,y,True)) for r,s in masks]
        conditional[a]=paired(unit,diff)
    diagnostic=[d for r in rows for s in r.get("diagnostics",{}).values() for d in s.values()]
    bounds={k:max(r[k] for r in diagnostic) for k in ("error","k_v_max","layer20_writes_max")} if diagnostic else {}
    return dict(n=len(rows),outcome=outcome,dual_other=du,contrasts=supports,upper_bounds=stops,
                compensation_counts=dict(helped=n_help,harmed=n_harm),different_reference=conditional,
                diagnostic=dict(pairs=sum("diagnostics" in r for r in rows),max_abs=bounds),gates=gates)
