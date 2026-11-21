"""Paired exact-integer route and content factorial comparisons."""

import numpy as np

from av_deck import DOMAINS
from edge_stats import upper_binomial

ARMS = ("route", "value", "joint", "minus_interaction", "minus_control", "identity")
DIRECTIONS = ("rescue", "damage")
PRIMARY = {"joint_route": ("joint","route"), "joint_value": ("joint","value"),
           "interaction": ("minus_control","minus_interaction")}
DESCRIPTIVE = {"route_identity": ("route","identity"), "value_identity": ("value","identity"),
               "control_cost": ("joint","minus_control")}
FAMILY = len(DOMAINS)*len(PRIMARY)


def contrast(x,y,primary):
    delta = np.asarray(x,dtype=np.int8) - np.asarray(y,dtype=np.int8)
    extra, loss = int((delta>0).sum()),int((delta<0).sum())
    rng = np.random.default_rng(261128231)
    values = np.concatenate([delta[rng.integers(len(delta),size=(1000,len(delta)))].mean(axis=1) for _ in range(40)])
    alpha = .025/FAMILY if primary else .025
    lo,hi = np.quantile(values,[alpha,1-alpha]).tolist()
    return dict(extra=extra,loss=loss,mean=float(delta.mean()),interval=[lo,hi],
                upper_97_5=upper_binomial(extra,len(delta)),upper_simultaneous=upper_binomial(extra,len(delta),alpha=.025/2))


def summarize(rows):
    result = {}
    for d in DOMAINS:
        q = [r for r in rows if r["domain"]==d]
        arms,dual = {},{}
        for a in ARMS:
            rescue = [r["pred"]["rescue"][a]["number"]==r["target"] for r in q]
            damage = [r["pred"]["damage"][a]["number"]==r["old"] for r in q]
            still = [r["pred"]["damage"][a]["number"]==r["target"] for r in q]
            dual[a] = [x and y for x,y in zip(rescue,damage,strict=True)]
            arms[a] = dict(rescue=sum(rescue),damage=sum(damage),dual=sum(dual[a]),k2_correct=sum(still))
        tests = {key:contrast(dual[a],dual[b],key in PRIMARY) for key,(a,b) in (PRIMARY|DESCRIPTIVE).items()}
        passed = min(arms[a]["dual"] for a in ("joint","minus_control")) >= 8
        result[d] = dict(n=len(q),arms=arms,contrasts=tests,
                         supported=passed and all(tests[k]["interval"][0] > 0 for k in PRIMARY))
    audit = [r["audit"] for r in rows if "audit" in r]
    status = dict(n=len(audit),checks=sum(a["checks"] for a in audit))
    status.update({k:max(a[k] for a in audit) for k in ("factorization","native_projection","suffix","identity")})
    return dict(domains=result,supported=all(r["supported"] for r in result.values()),
                stop=all(r["contrasts"]["interaction"]["upper_simultaneous"]<.10 for r in result.values()),audit=status)
