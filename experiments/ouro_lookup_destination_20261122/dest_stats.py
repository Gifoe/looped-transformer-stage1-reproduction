"""Exact destination-specific rescue/reversal on fixed independent cohorts."""

import numpy as np

from dest_deck import DOMAINS
from edge_stats import upper_binomial

DIRECTIONS=("rescue","damage")
ARMS=("true","redirect","live_full","identity")


def contrast(x):
    if not x:
        return None
    a=np.asarray(x,dtype=np.int8)
    rng=np.random.default_rng(261132229)
    sample=np.concatenate([a[rng.integers(len(a),size=(1000,len(a)))].mean(1) for _ in range(40)])
    p=.025/6
    return dict(n=len(a),mean=float(a.mean()),extra=int((a>0).sum()),loss=int((a<0).sum()),
                interval=np.quantile(sample,[p,1-p]).tolist())


def classify(row,arm):
    x,y=(row["pred"][d][arm]["number"] for d in DIRECTIONS)
    t,f,u=(row[k] for k in ("target","foil","third_value"))
    return dict(rescue=int(x==t),reversal=int(y==f),dual=int(x==t and y==f),
                third_rescue=int(x==u),third_damage=int(y==u),third_dual=int(x==u and y==u),
                other_rescue=int(x not in (t,f,u)),other_damage=int(y not in (t,f,u)))


def summarize(rows,cohort):
    idx={r["id"]:r for r in rows}
    summary={}
    for d in DOMAINS:
        b=[r for r in rows if r["domain"]==d]
        subsets={"repair":[r for r in b if r["kind"]=="repair"],
                 "projected":[r for r in b if r["kind"]=="repair" and r["projected_match"]],
                 "stable":[r for r in b if r["kind"]=="stable"]}
        calls={r["id"]:{a:classify(r,a) for a in ARMS} for r in b}
        counts={}
        for k,s in subsets.items():
            counts[k]=dict(n=len(s),arms={})
            for a in ARMS:
                keys=classify(b[0],a)
                counts[k]["arms"][a]={p:sum(calls[r["id"]][a][p] for r in s) for p in keys}
        delta={r["id"]:calls[r["id"]]["true"]["dual"]-calls[r["id"]]["redirect"]["dual"] for r in b}
        direct=contrast([delta[r["id"]] for r in subsets["repair"]])
        direct["upper"]=upper_binomial(direct["extra"],direct["n"],alpha=.025/2)
        proj=contrast([delta[r["id"]] for r in subsets["projected"]])
        pairs=cohort[d]
        spec=contrast([delta[idx[p["repair"]]["id"]]-delta[idx[p["stable"]]["id"]] for p in pairs])
        audit=[r["audit"] for r in b if "audit" in r]
        summary[d]=dict(counts=counts,direct=direct,projected=proj,specificity=spec,
                        supported=bool(direct["n"]>=48 and proj is not None and proj["n"]>=48 and len(pairs)>=48 and
                                       counts["repair"]["arms"]["true"]["dual"]>=8 and
                                       all(x["interval"][0]>0 for x in (direct,proj,spec))),
                        audit=dict(n=len(audit),checks=sum(r["checks"] for r in audit),
                                   maxima={k:max((r[k] for r in audit),default=0.) for k in ("debit","l2","probability","attention")}))
    return dict(domains=summary,supported=all(x["supported"] for x in summary.values()),
                excluded=all(x["direct"]["upper"]<.10 for x in summary.values()))
