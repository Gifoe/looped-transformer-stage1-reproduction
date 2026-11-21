"""Prespecified exact-answer patterns and clustered exclusions."""
from edge_stats import upper_binomial
from origin_stats import paired, positive

SCOPES=("local","full")
TYPES=("authentic","geometry")
BRANCHES={"attention":("a_write",),"mlp":("m_write",),"both":("a_write","m_write")}
CONFIG={f"{s}_{t}_{k}":(s,t,v) for s in SCOPES for t in TYPES for k,v in BRANCHES.items()}
ARMS=("native",*CONFIG)
GROUPS=("original","paired_natural_repair","paired_correct_k1","paired_other_k1_error")
PATTERNS=("attention_dual","mlp_dual","joint_only")


def outcomes(r,s,t):
    pred=r["pred"]
    a,m,b=[pred[f"{s}_{t}_{k}"]["number"] for k in BRANCHES]
    y,w=r["target"],r["foil"]
    old=a==w and w!=y
    back=m==w and w!=y
    base=b==w and w!=y
    return dict(attention_loss=int(old),mlp_loss=int(back),both_loss=int(base),
                attention_restore=int(base and m==y),mlp_restore=int(base and a==y),
                attention_dual=int(old and base and m==y),mlp_dual=int(back and base and a==y),
                joint_only=int(base and a!=y and m!=y),
                redundancy=int(base and a==y and m==y),
                signed_interaction=1-int(a==y)-int(m==y)+int(b==y))


def summarize(rows):
    view={}
    for group in GROUPS:
        q=[r for r in rows if r["group"]==group]
        d=dict(n=len(q),arms={},scopes={})
        for arm in ARMS:
            ys=[r["pred"][arm]["number"] for r in q]
            d["arms"][arm]=dict(correct=sum(y==r["target"] for y,r in zip(ys,q,strict=True)),
                                 wrong_k1=sum(y==r["foil"] and y is not None and y!=r["target"] for y,r in zip(ys,q,strict=True)),
                                 non_integer=sum(y is None for y in ys))
        if group not in ("original","paired_natural_repair") or not q:
            view[group]=d
            continue
        for s in SCOPES:
            one,two=[[outcomes(r,s,t) for r in q] for t in TYPES]
            counts={t:{k:sum(x[k] for x in v) for k in v[0]} for t,v in zip(TYPES,(one,two),strict=True)}
            effects={}
            bounds={}
            gates={}
            stoppers={}
            for k in one[0]:
                score=[a[k]-b[k] for a,b in zip(one,two,strict=True)]
                effects[k]=paired(q,score)
                if k in PATTERNS:
                    extra=sum(x>0 for x in score)
                    upper=upper_binomial(extra,len(q))
                    bounds[k]=dict(n=len(q),events=extra,upper_97_5=upper)
                    gates[k]=extra>=8 and positive(effects[k])
                    stoppers[k]=upper<.10
            d["scopes"][s]=dict(totals=counts,paired=effects,bounds=bounds,gates=gates,exclude_large=stoppers)
        view[group]=d
    gate={s:view["original"]["scopes"][s]["gates"] for s in SCOPES}
    exclude={s:view["original"]["scopes"][s]["exclude_large"] for s in SCOPES}
    diag=[v for r in rows for v in r.get("additive_error",{}).values()]
    return dict(groups=view,gates=gate,exclude_large=exclude,
                close_late_suffix=all(x for q in exclude.values() for x in q.values()),
                diagnostic=dict(n=sum("additive_error" in r for r in rows),max_abs=max(diag,default=None)))
