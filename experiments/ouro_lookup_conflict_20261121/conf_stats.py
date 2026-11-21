"""Paired whole-answer switch rates, calibrated controls, and exclusion bounds."""

import numpy as np

from conf_deck import DOMAINS
from edge_stats import upper_binomial

DIRECTIONS = ("original","swapped")
PATHS = ("value","carry")
CONTROLS = ("geometry","unrelated")
ARMS = ("value","value_geometry","value_unrelated","carry","carry_geometry","carry_unrelated","both","clean")
FAMILY = len(DOMAINS)*len(PATHS)*len(CONTROLS)


def contrast(x,y,primary):
    if not x:
        return None
    d = np.asarray(x,dtype=np.int8)-np.asarray(y,dtype=np.int8)
    wins,losses = int((d>0).sum()),int((d<0).sum())
    rng = np.random.default_rng(261130179)
    z = np.concatenate([d[rng.integers(len(d),size=(1000,len(d)))].mean(1) for _ in range(40)])
    a = .025/FAMILY if primary else .025
    return dict(n=len(d),extra=wins,loss=losses,mean=float(d.mean()),interval=np.quantile(z,[a,1-a]).tolist(),
                upper=upper_binomial(wins,len(d),alpha=.025/FAMILY) if primary else None)


def analyze(q):
    counts,dual = {},{}
    for a in ARMS:
        swaps = []
        per = {}
        for s in DIRECTIONS:
            r = "target" if s=="original" else "old"
            drop = "old" if s=="original" else "target"
            transfer = [p["pred"][s][a]["number"]==p[drop] for p in q]
            preserve = [p["pred"][s][a]["number"]==p[r] for p in q]
            swaps.append(transfer)
            per[s] = dict(donor=sum(transfer),recipient=sum(preserve),other=len(q)-sum(transfer)-sum(preserve))
        dual[a] = [x and y for x,y in zip(*swaps,strict=True)]
        counts[a] = dict(**per,dual=sum(dual[a]))
    return counts,dual


def summarize(rows):
    results = {}
    for d in DOMAINS:
        out = {}
        for g in ("repair","stable"):
            cases = [r for r in rows if r["domain"]==d and r["kind"]==g]
            counts,dual = analyze(cases)
            out[g] = dict(n=len(cases),counts=counts,value_vs_carry=contrast(dual["value"],dual["carry"],False))
            if g=="repair":
                tests = {p:{c:contrast(dual[p],dual[f"{p}_{c}"],True) for c in CONTROLS} for p in PATHS}
                out[g]["tests"] = tests
                out[g]["supported"] = [p for p in PATHS if counts[p]["dual"]>=8 and all(x["interval"][0]>0 for x in tests[p].values())]
                out[g]["excluded"] = [p for p in PATHS if all(x["upper"]<.10 for x in tests[p].values())]
        results[d] = out
    audit = [r["audit"] for r in rows if "audit" in r]
    v = dict(n=len(audit),tokens=sum(r["tokens"] for r in audit),v_sites=sum(r["v_sites"] for r in audit),
             carry_sites=sum(r["carry_sites"] for r in audit))
    v.update({k:max(r[k] for r in audit) for k in ("cache_causal","attention")})
    return dict(domains=results,supported=[p for p in PATHS if all(p in results[d]["repair"]["supported"] for d in DOMAINS)],
                excluded=[p for p in PATHS if all(p in results[d]["repair"]["excluded"] for d in DOMAINS)],audit=v)
