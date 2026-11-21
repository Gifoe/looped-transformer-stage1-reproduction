"""Predeclared simultaneous contrasts and conservative interval localization."""
import numpy as np

from edge_stats import upper_binomial
from onset_deck import DOMAINS

SITES=(1,6,12,18,20)
MODES={"phase":("authentic","geometry"),"content":("authentic","geometry","unrelated")}
FAMILY=30
DRAWS=40000


def ncompare(delta):
    a=np.asarray(delta,dtype=np.float64)
    if a.size==0:
        return dict(n=0,total=0,mean=None,simultaneous=[None,None],extra=0,upper_97_5=1.,upper_across_sites=1.)
    rng=np.random.default_rng(261125103)
    estimates=[]
    for start in range(0,DRAWS,1000):
        n=min(1000,DRAWS-start)
        estimates.extend(a[rng.integers(len(a),size=(n,len(a)))].mean(axis=1).tolist())
    a0=.05/(2*FAMILY)
    lo,hi=np.quantile(estimates,[a0,1-a0]).tolist()
    e=int((a>0).sum())
    return dict(n=len(a),total=int(a.sum()),mean=float(a.mean()),simultaneous=[lo,hi],extra=e,
                upper_97_5=upper_binomial(e,len(a)),upper_across_sites=upper_binomial(e,len(a),alpha=.025/5))


def summarize(rows):
    out={}
    for l in SITES:
        a={}
        for domain in DOMAINS:
            b={}
            for k,modes in MODES.items():
                q=[r for r in rows if r["domain"]==domain and k in r["pred"][str(l)]]
                effects={}
                counts={}
                dual={}
                for mode in modes:
                    f=[]; r0=[]
                    for r in q:
                        forward,reverse=[r["pred"][str(l)][k][d][mode]["number"] for d in ("forward","reverse")]
                        f.append(forward==r["target" if k=="phase" else "old"])
                        r0.append(reverse==r["old" if k=="phase" else "target"])
                    dual[mode]=[int(x and y) for x,y in zip(f,r0,strict=True)]
                    counts[mode]=dict(forward=sum(f),reverse=sum(r0),dual=sum(dual[mode]))
                for m in modes[1:]:
                    effects[m]=ncompare([x-y for x,y in zip(dual["authentic"],dual[m],strict=True)])
                passes=counts["authentic"]["dual"]>=8 and all(v["simultaneous"][0]>0 for v in effects.values())
                b[k]=dict(n=len(q),counts=counts,contrasts=effects,supported=passes)
            a[domain]=b
        out[str(l)]=a
    def supported(l,kinds):
        return all(out[str(l)][d][k]["supported"] for d in DOMAINS for k in kinds)
    reference=supported(20,("content",))
    first=next((l for l in SITES if supported(l,MODES)),None) if reference else None
    previous=SITES[SITES.index(first)-1] if first is not None and first!=1 else None
    maximum={}
    if previous is not None:
        earlier=[v for d in DOMAINS for k in MODES for v in out[str(previous)][d][k]["contrasts"].values()]
        maximum={k:max(v[k] for v in earlier) for k in ("upper_97_5","upper_across_sites")}
    bracket=bool(maximum) and max(maximum.values())<.10
    audit=[z["parity"] for z in rows if "parity" in z]
    return dict(layers=out,decision=dict(reference=reference,first=first,preceding=previous,
                                         preceding_maximum=maximum,identified_interval=bracket),
                diagnostic=dict(records=len(audit),checks=sum(r["n"] for r in audit),
                                max_abs=max((r["max_abs"] for r in audit),default=None)))
