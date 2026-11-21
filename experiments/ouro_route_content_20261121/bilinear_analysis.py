"""Separate true operator-product and downstream-nonlinearity effects."""
import argparse
from pathlib import Path

from analysis import metric,scores,winner
from native import load,save,sha


def summarize(raw,meta):
    result={}
    for key in ("L6_attention","L1_mlp"):
        cases=[]
        for row,m in zip(raw,meta,strict=True):
            assert row["id"]==m["id"]
            a=row["cases"][key]
            g=m["gold"]
            early=scores(row["K1"])
            late=scores(a["native_ll"])
            fixed=max((j for j in range(len(late)) if j!=g),key=lambda j:late[j])
            q={k:scores(v["ll"]) for k,v in a["branches"].items()}
            margins={k:z[g]-z[fixed] for k,z in q.items()}
            n=late[g]-late[fixed]
            product=margins["early_both"]-margins["additive"]
            nonlinear=margins["additive"]-margins["early_first"]-margins["early_second"]+n
            ix=margins["early_both"]-margins["early_first"]-margins["early_second"]+n
            assert abs(ix-product-nonlinear)<1e-10
            cases.append(dict(pop="".join("C" if winner(x)==g else "W" for x in (early,late)),
                 product=product,downstream=nonlinear,total=ix,
                 product_vs_same_norm=margins["early_both"]-margins["permuted"],
                 ratio=a["relative_operator_product"],noop=a["noop_error"],resid=a["algebra_residual"],
                 hit_both=winner(q["early_both"])==g,hit_add=winner(q["additive"])==g,
                 hit_control=winner(q["permuted"])==g))
        groups={}
        for p in ("ALL","WC","CC","CW","WW"):
            z=cases if p=="ALL" else [x for x in cases if x["pop"]==p]
            groups[p]={k:metric([x[k] for x in z]) for k in ("product","downstream","total","product_vs_same_norm","ratio")}
            groups[p].update(n=len(z),hits={k:sum(x[k] for x in z) for k in ("hit_both","hit_add","hit_control")})
        result[key]=dict(groups=groups,operator_identity_max=max(x["resid"] for x in cases),
                        noop_max=max(x["noop"] for x in cases))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    o=p.parse_args().out
    m=load(o/"NATURAL_MANIFEST.json")["cohorts"]
    result={}
    for k in m:
        x=load(o/f"BILINEAR_{k}.json")
        assert len(x["rows"])==x["expected"]==128
        assert x["manifest_sha256"]==sha(o/"NATURAL_MANIFEST.json")
        result[k]=summarize(x["rows"],m[k]["rows"][:128])
    save(o/"BILINEAR_SUMMARY.json",result)

