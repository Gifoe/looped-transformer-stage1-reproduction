"""Directed native K2 deletion effects and matched stable specificity."""

import numpy as np

from edge_stats import upper_binomial
from nec_deck import DOMAINS
from nec_match import MIN_MATCHES

ARMS = ("true", "geometry", "identity")
FAMILY = 4


def endpoint(row,arm,foil):
    pred = row["pred"][arm]["number"]
    return int(pred==row["foil"]) if foil else int(pred!=row["target"])


def delta(row,foil=True):
    return endpoint(row,"true",foil)-endpoint(row,"geometry",foil)


def contrast(diffs,primary):
    if not diffs:
        return None
    x = np.asarray(diffs,dtype=np.int8)
    rng = np.random.default_rng(261129127)
    draw = np.concatenate([x[rng.integers(len(x),size=(1000,len(x)))].mean(axis=1) for _ in range(40)])
    p = .025/FAMILY if primary else .025
    return dict(n=len(x),mean=float(x.mean()),positive=int((x>0).sum()),negative=int((x<0).sum()),
                interval=np.quantile(draw,[p,1-p]).tolist())


def summarize(rows,match):
    ids = {r["id"]:r for r in rows}
    out = {}
    for domain in DOMAINS:
        r,s = [[q for q in rows if q["domain"]==domain and q["kind"]==kind] for kind in ("repair","stable")]
        arms = {}
        for kind,pool in (("repair",r),("stable",s)):
            elig = [q for q in pool if q["foil"] is not None]
            counts = {a:dict(foil=sum(endpoint(q,a,True) for q in elig),error=sum(endpoint(q,a,False) for q in pool)) for a in ARMS}
            arms[kind] = dict(n=len(pool),foil_n=len(elig),counts=counts)
        directed = contrast([delta(q) for q in r],True)
        directed["upper_simultaneous"] = upper_binomial(directed["positive"],len(r),alpha=.025/len(DOMAINS))
        pairs = match[domain]
        specificity = contrast([delta(ids[p["repair"]])-delta(ids[p["stable"]]) for p in pairs],True)
        stable_foil = contrast([delta(q) for q in s if q["foil"] is not None],False)
        stable_error = contrast([delta(q,False) for q in s],False)
        repair_error = contrast([delta(q,False) for q in r],False)
        gaps = [p["absolute_gap"] for p in pairs]
        balance = dict(n=len(gaps),mean=float(np.mean(gaps)) if gaps else None,max=max(gaps,default=None))
        success = arms["repair"]["counts"]["true"]["foil"]>=8 and directed["interval"][0]>0
        support = bool(success and balance["n"]>=MIN_MATCHES and specificity["interval"][0]>0)
        out[domain] = dict(arms=arms,directed=directed,specificity=specificity,stable_foil=stable_foil,
                           stable_error=stable_error,repair_error=repair_error,balance=balance,supported=support)
    audit = [r["audit"] for r in rows if "audit" in r]
    check = dict(n=len(audit),checks=sum(a["checks"] for a in audit))
    check.update({k:max(a[k] for a in audit) for k in ("factorization","suffix","identity","subtraction")})
    return dict(domains=out,supported=all(q["supported"] for q in out.values()),
                excludes_ten=all(q["directed"]["upper_simultaneous"]<.10 for q in out.values()),audit=check)
