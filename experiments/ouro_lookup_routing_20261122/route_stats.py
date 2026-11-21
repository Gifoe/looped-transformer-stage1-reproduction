"""Exact pair-level rollback, rescue and margin-matched specificity."""

import numpy as np

from edge_stats import upper_binomial
from route_deck import DOMAINS

DIRECTIONS = ("rescue","damage")
PRIMARY = ("route","value","joint")
ARMS = (*PRIMARY,"route_control","value_control","joint_control","route_full","joint_full","identity")
FAMILY = len(PRIMARY)*len(DOMAINS)*2


def contrast(x):
    if not x:
        return None
    a = np.asarray(x,dtype=np.int8)
    rng = np.random.default_rng(261131223)
    draw = np.concatenate([a[rng.integers(len(a),size=(1000,len(a)))].mean(1) for _ in range(40)])
    p = .025/FAMILY
    return dict(n=len(a),mean=float(a.mean()),extra=int((a>0).sum()),loss=int((a<0).sum()),interval=np.quantile(draw,[p,1-p]).tolist())


def outcomes(r,arm):
    p,q = (r["pred"][d][arm]["number"] for d in DIRECTIONS)
    return int(p==r["target"]),int(q==r["foil"]),int(q!=r["target"])


def summarize(rows,pairs):
    index = {r["id"]:r for r in rows}
    z = {}
    for d in DOMAINS:
        cases = [r for r in rows if r["domain"]==d]
        p = [r for r in cases if r["kind"]=="repair"]
        stable = [r for r in cases if r["kind"]=="stable"]
        d0 = {}
        for arm in ARMS:
            local={}
            for kind,cohort in (("repair",p),("stable",stable)):
                a = [outcomes(r,arm) for r in cohort]
                local[kind] = dict(n=len(a),rescue=sum(x for x,_,_ in a),reversal=sum(y for _,y,_ in a),
                                   error=sum(v for _,_,v in a),dual=sum(x*y for x,y,_ in a))
            d0[arm]=local
        tests = {}
        for arm in PRIMARY:
            def dual(r,a):
                x,y,_ = outcomes(r,a)
                return x*y
            diffs = {r["id"]:dual(r,arm)-dual(r,arm+"_control") for r in cases}
            direct = contrast([diffs[r["id"]] for r in p])
            direct["upper"] = upper_binomial(direct["extra"],len(p),alpha=.025/(FAMILY//2))
            spec = contrast([diffs[b["repair"]]-diffs[b["stable"]] for b in pairs[d]])
            tests[arm] = dict(directed=direct,specificity=spec,supported=len(p)>=48 and len(pairs[d])>=48 and
                              d0[arm]["repair"]["dual"]>=8 and direct["interval"][0]>0 and spec["interval"][0]>0,
                              excluded=direct["upper"]<.10)
        q=[r["audit"] for r in cases if "audit" in r]
        z[d]=dict(counts=d0,tests=tests,audit=dict(n=len(q),checks=sum(r["checks"] for r in q),
                  errors={k:max((r[k] for r in q),default=0.) for k in ("mass","write","attention")}))
    return dict(domains=z,supported=[a for a in PRIMARY if all(z[d]["tests"][a]["supported"] for d in DOMAINS)],
                excluded=[a for a in PRIMARY if all(z[d]["tests"][a]["excluded"] for d in DOMAINS)])
